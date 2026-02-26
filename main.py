import os
import hydra
from dotenv import load_dotenv
import torch

from dataset import CoresetDataset
from transformers import  Qwen2_5_VLForConditionalGeneration


from helpers.k_sampling import k_sampling
from helpers.score import optimize_fixed_k_classwise
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="./config/CRC100K/gauc", config_name="zero_shot", version_base="1.3") # TODO: change config_path to your config folder
def main(cfg):
    load_dotenv()
    
    data_cfg = cfg.data

    if data_cfg.num_shots == 0:
        assert (
            data_cfg.show_bbox == False
        ), "show_bbox can only be used with num_shots > 0"

    dataset = CoresetDataset(
            datafile_path=data_cfg.datafile_path,
            use_only=data_cfg.use_only,
            label_replacements=data_cfg.label_replacements,
            dataset_vectors_path=data_cfg.dataset_vectors_path,
            use_tiles=data_cfg.use_tiles if hasattr(data_cfg, "use_tiles") else False,
            most_similar_last=data_cfg.most_similar_last if hasattr(data_cfg, "most_similar_last") else False,
            take_random=data_cfg.take_random if hasattr(data_cfg, "take_random") else False,

    )

    dataset.pprint_self()
    print("# dataframe entries: ", len(dataset.data))

    # change your models here
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model) 
    model.eval().tie_weights()

    selected_imgs = k_sampling(cfg.data.dataset_vectors_pathpath)

    k = 3
    optimize_fixed_k_classwise(selected_imgs, k,model, dataset, cfg)



if __name__ == "__main__":
    main()

