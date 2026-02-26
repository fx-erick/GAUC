import json
import os

import fsspec
import random
import torch
import torch.nn.functional as F

import numpy as np
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from base64 import b64encode
from emid import EMI, EMIDupperbound


def flatten(S_dict):
        classes = list(S_dict.keys())

        return [img for cls in classes for img in S_dict[cls]]

def optimize_fixed_k_classwise(
    candidates,
    k_per_class,
    model,
    dataset,
    cfg,
    init_S=None,
    num_steps=1000,
    temperature=0.0,
    verbose=True,
):
    

    classes = list(candidates.keys())

    if init_S is None:
        S = {
            cls: random.sample(candidates[cls], k_per_class)
            for cls in classes
        }
    else:
        S = {cls: list(v) for cls, v in init_S.items()}

    

    best_S = {cls: list(v) for cls, v in S.items()}
    dataset.candidates = S



    emi_estimator = EMI(feature_dim=cfg.emi.feature_dim,
                        mi_est_dim=cfg.emi.mi_est_dim,
                        mi_ckpt_path=cfg.emi.mi_ckpt_path,
                        v_embedder_name=cfg.emi.v_embedder_name,
                        t_embedder_name=cfg.emi.t_embedder_name,)

    best_score = score_fn(model, candidates,S, dataset, emi_estimator, cfg)

    # ---- optimization loop ----
    for step in range(num_steps):
        cls = random.choice(classes)

        i = random.randrange(k_per_class)
        pool = candidates[cls]
        new_img = random.choice(pool)

        if new_img in S[cls]:
            continue 

        # 4) construct proposal
        S_new = {c: list(v) for c, v in S.items()}
        S_new[cls][i] = new_img

        # 5) evaluate
        score_new = score_fn(model, candidates, S_new, dataset, emi_estimator, cfg)
        delta = score_new - best_score

        # 6) accept / reject
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-8)):
            S = S_new
            if score_new < best_score:
                best_S = {c: list(v) for c, v in S_new.items()}
                best_score = score_new
                dataset.candidates = S


        if verbose and step % 5 == 0:
            print(f"[{step:04d}] best score = {best_score:.6f}")
        
        print(score_new)

    return best_S, best_score


def rbf_kernel(X, Y, sigma):
    """
    X: (N, D)
    Y: (M, D)
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    sq_dist = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-sq_dist / (2 * sigma**2))


def median_heuristic_sigma(X):
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    return np.median(dists[dists > 0])

def get_embeddings(files,cfg):
    if isinstance(files, list):
        paths = files
    else:
        paths = np.array(list(files.values())).flatten()

    path = cfg.data.dataset_vectors_pathpath
    feat_array = np.load(path, allow_pickle=True)
    feat_dict = {row['name']: row['vector'] for row in feat_array}
    vectors = np.array([
        feat_dict[name]
        for name in paths
    ])

    return vectors


def mmd_squared(X_full, Y_sel,cfg):

    X_full = get_embeddings(X_full,cfg)
    Y_sel = get_embeddings(Y_sel,cfg)
    sigma = median_heuristic_sigma(X_full)
    
    K_xx = rbf_kernel(X_full, X_full, sigma)
    K_yy = rbf_kernel(Y_sel, Y_sel, sigma)
    K_xy = rbf_kernel(X_full, Y_sel, sigma)

    mmd2 = (
        K_xx.mean()
        + K_yy.mean()
        - 2 * K_xy.mean()
    )
    return float(mmd2)


def score_fn(model, candidates , S, dataset, emi_estimator,cfg):
        
 
    mmd_perf = mmd_squared(candidates, flatten(S), cfg)
    shift_perf = mi_estimator(model,dataset, emi_estimator,cfg)

    eval_subset = []
    seed = 42
    rng = random.Random(seed)

    multi_shot_mappings = dataset.get_multi_shot_mappings()
        
    multi_shot_mappings = {
            key: [encode_image(str(p)) for p in paths]
            for key, paths in multi_shot_mappings.items()
    }


    subset_size = len(dataset.data)
    eval_indices = rng.sample(range(len(dataset.data)), subset_size)
    for idx in eval_indices:
        sample = list(dataset(num_shots=dataset.candidates is not None))[idx]
        eval_subset.append(sample)

    
    variance = var_estimator(model, eval_subset, multi_shot_mappings,cfg)


        
    return  mmd_perf + cfg.lambda_shift * shift_perf + cfg.lambda_var * variance 



def mi_estimator(model,dataset, emi_estimator,cfg):

    query_img_path = "None" #example query image
    query_img = encode_image(str(query_img_path))
    query_img=[query_img] if isinstance(query_img, str) else query_img
    gt_label = os.path.basename(query_img_path).split('-')[0]

    system_prompt_path = cfg.user_args.system_prompt_path
    user_query_path = cfg.user_args.user_query_path
    with fsspec.open(system_prompt_path, mode="r") as f:
        system_prompt = f.read()

    with fsspec.open(user_query_path, mode="r") as f:
        user_query = f.read()

    multi_shot_mappings = dataset.get_multi_shot_mappings()
  
    multi_shot_mappings = {
        key: [encode_image(str(p)) for p in paths]
        for key, paths in multi_shot_mappings.items()
    }

    response, emi, probs = get_response(model, system_prompt, user_query, query_img, multi_shot_mappings, gt_label, emi_estimator,cfg)

    
    system_prompt_para_path = cfg.user_args.system_prompt_para_path
    user_query_para_path = cfg.user_args.user_query_para_path
    with fsspec.open(system_prompt_para_path, mode="r") as f:
        system_prompt_para = f.read()

    with fsspec.open(user_query_para_path, mode="r") as f:
        user_query_para = f.read()


    response_para, emi_para, probs_para = get_response(model,system_prompt_para, user_query_para, query_img, multi_shot_mappings, gt_label, emi_estimator)
    p_zv, p_zt, p_zyh, p_zy = probs
    q_zv, q_zt, q_zyh, q_zy = probs_para
    emid_ub = EMIDupperbound(p_zv, p_zt, p_zyh, p_zy, q_zv, q_zt, q_zyh, q_zy, None)

    return emid_ub[0]

def var_estimator(model, eval_subset, multi_shot_mappings,cfg):
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model_name)

    system_prompt_path = cfg.user_args.system_prompt_path
    user_query_path = cfg.user_args.user_query_path
    with fsspec.open(system_prompt_path, mode="r") as f:
        system_prompt = f.read()

    with fsspec.open(user_query_path, mode="r") as f:
        user_query = f.read()

    class_labels = cfg.class_labels
        
        # Pre-compute token IDs for your classes (handles multi-token gracefully)
    class_token_ids = []
    for label in class_labels:
        tokens = processor.tokenizer.encode(label, add_special_tokens=False)
        class_token_ids.append(tokens[0])

    for sample in eval_subset:
        image_path = sample["image_path"]
        query_img = encode_image(str(image_path))
        query_img=[query_img] if isinstance(query_img, str) else query_img
        

       
        messages = multi_shot(system_prompt,user_query, query_img, multi_shot_mappings)


        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [img.resize((224, 224)) for img in image_inputs]
    
    
    
        inputs = processor(
            text=[text],         # your original prompt text
            images=image_inputs, # list of PIL images
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")



        # --- forward pass with teacher forcing ---
        with torch.no_grad():
            outputs = model(**inputs)

            first_token_logits = outputs.logits[:, -1, :]

        class_logits = first_token_logits[0, class_token_ids]  # Shape: (num_classes,)
        
        class_logprobs = F.log_softmax(class_logits, dim=-1)
        total_variance = torch.var(class_logprobs)

        
    return total_variance.float().cpu().numpy()



def get_response(model, system_prompt, user_query, query_img, multi_shot_mappings, gt_label, emi_estimator, cfg):

    processor = AutoProcessor.from_pretrained(cfg.model_name)

    messages = multi_shot(system_prompt,user_query, query_img, multi_shot_mappings)
    ideal_responses_json = "None"
    with open(ideal_responses_json) as f:
        ideal_responses = json.load(f)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    image_inputs = [img.resize((224, 224)) for img in image_inputs]
    
    
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0,               
    do_sample=False , output_scores=True, return_dict_in_generate=True  )


    generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
    ]

    resp_model = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    resp_model = clean_json_response(resp_model[0]) 

    resp_base = ideal_responses[gt_label]
    v_queries = image_inputs
    t_queries = text

    
    resp_model = json.loads(resp_model)
    
    
    emi_outputs = emi_estimator(v_queries, t_queries, resp_model["thoughts"], resp_base, True)
    
    
    emi_vals = emi_outputs[0]
    probs = emi_outputs[3:]    

    return resp_model, emi_vals, probs

def _gen_system_message(system_prompt: str) :
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
            },
        ],
    }

def _gen_user_message(user_query: str, images):
    assert isinstance(
        images, list
    ), "images must be a list, for single image use [image]"

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_query,
            },
            *[
                {
                    "type": "image",
                    "image": 
                         f"data:image/jpeg;base64,{image}"
                }
                for image in images
            ],
        ],
    }




def multi_shot(
        system_prompt: str,
        user_query: str,
        query_images,
        multi_shot_mappings,
    ):
        messages = [_gen_system_message(system_prompt)]

        # TODO: find a nicer way to split the user query
        user_query_pre = user_query.split("-----------")[0].strip()
        user_query_post = user_query.split("-----------")[1].strip()

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query_pre,
                    }
                ],
            }
        )


        max_len = len(list(multi_shot_mappings.values())[0])
        for i in range(max_len):
            for j, instruct in enumerate(multi_shot_mappings.keys()):
                if i < len(multi_shot_mappings[instruct]):
                    image = multi_shot_mappings[instruct][i]
                    image_content = {
                        "type": "image",
                        "image": 
                         f"data:image/jpeg;base64,{image}"
                    }
                    examples = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruct},
                            image_content,
                        ],
                    }
                    messages.append(examples)


        messages.append(_gen_user_message(user_query_post, query_images))
        return messages



def encode_image(image_path) -> str:
        with fsspec.open(image_path, mode="rb") as img_file:
            return b64encode(img_file.read()).decode("utf-8")

def batch_encode_images( image_paths ) -> None: # TODO improve type hints
    
    
    encoded_images = {}
    for image_path in image_paths:
        encoded_images[image_path] = encode_image(image_path)
    return encoded_images



def clean_json_response(response):
    """
    Processes the VLM response to ensure it's in the proper JSON format.
    Removes ```json and ``` markers if present.
    """
    if isinstance(response, list):
        # Get the first item if it's a list
        response_str = response[0]
        
        # Remove ```json and ``` markers
        if response_str.startswith('```json') and response_str.endswith('```'):
            # Remove both markers and strip whitespace
            cleaned = response_str[7:-3].strip()
            return [cleaned]
        if response_str.startswith('```json') and response_str.endswith('``` '):
            # Remove both markers and strip whitespace
            cleaned = response_str[7:-4].strip()
            cleaned = cleaned.lstrip('\ufeff')
            return [cleaned]
        elif response_str.startswith('```') and response_str.endswith('```'):
            # Handle case where it's just ``` without json
            cleaned = response_str[3:-3].strip()
            return [cleaned]
    
    # Return original if no cleaning needed
    return response