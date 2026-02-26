from transformers import CLIPModel, CLIPProcessor
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import torch.nn.functional as F

class CLUB(torch.nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size//2),
                                    torch.nn.Tanh(),
                                    torch.nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size//2),
                                    torch.nn.Tanh(),
                                   torch.nn.Linear(hidden_size//2, y_dim),
                                    torch.nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)

        positive = - (mu - y_samples)**2 / (2. * logvar.exp())
        positive = positive.sum(dim=-1)

        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)

        negative = - ((y_samples_1 - prediction_1)**2) / (2. * logvar.exp().unsqueeze(1))
        negative = negative.sum(dim=-1)
        negative = negative.mean(dim=1)

        mi = (positive - negative).mean()
  

        return mi
    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
   


#! JSD Estimator
'''
Source: "https://github.com/uk-cliplab/representationJSD/blob/main/Neural_estimation/jsd_estimators.py", 
'''
def vonNeumannEntropy(K, lowRank = False, rank = None):
    n = K.shape[0]
    ek, _ = torch.linalg.eigh(K)
    if lowRank:
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        remainder = ek.sum() - ek_lr.sum()
        ek_lr[:(n-rank)] = remainder/(n-rank)
        mk = torch.gt(ek_lr, 0.0)
        mek = ek_lr[mk]
    else:
        mk = torch.gt(ek, 0.0)
        mek = ek[mk]

    mek = mek/mek.sum()   
    H = -1*torch.sum(mek*torch.log(mek))
    return H

def deep_JSD(X,Y,model):
    phiX = model(X)
    phiY = model(Y)
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

def JSD_cov(covX,covY):
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD



class EMI(torch.nn.Module):
    def __init__(self,
                 feature_dim=768,
                 mi_est_dim=500,
                 mi_ckpt_path=None,      # "estimator_ckpt/CLUB_synthetic.pt"
                 v_embedder_name=None,   # "openai/clip-vit-base-patch32"
                 t_embedder_name=None,   # "xlm-roberta-base" 
                 ):
        super(EMI, self).__init__()
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        self.mi_est = CLUB(feature_dim, feature_dim, mi_est_dim).to(self.device)
        print(f"\nInitialize MI estimator working on {feature_dim}-dim inputs...\n"
              f"Your embeddings of X, Y should be {feature_dim}-dim\n")

        if mi_ckpt_path is not None:
            print(f"Load (pre-)trained MI estimator from {mi_ckpt_path}...")
            self.mi_est.load_state_dict(torch.load(mi_ckpt_path, map_location=self.device))
        else:
            print("MI estimator needs to be trained first!!!")
        
        if (v_embedder_name is not None) and (t_embedder_name is not None):
            print(f"\nMI estimator will work on raw image and text directly with\n"
                  f"{v_embedder_name} and {t_embedder_name} as encoder models.\n")
            self.v_embedder = CLIPModel.from_pretrained(v_embedder_name).to(self.device) # "openai/clip-vit-base-patch32"
            self.v_processor = CLIPProcessor.from_pretrained(v_embedder_name)
            self.t_embedder = XLMRobertaModel.from_pretrained(t_embedder_name).to(self.device) # "xlm-roberta-base" 
            self.t_processor = XLMRobertaTokenizer.from_pretrained(t_embedder_name)
        else:
            print(f"MI estimator will work on pre-extracted embeddings")
            self.v_embedder = None
            self.v_processor = None
            self.t_embedder = None
            self.t_processor = None
        
    def forward(self,x_v,x_t,y_hat,y, return_emb=True):
        '''
        expected format: 
        (1) list of RGB images + list of raw text (default)
        (2) 2D tensor of pre-extraced img/txt embeddings
        '''
        if (self.v_embedder is not None) and (self.t_embedder is not None):
            v_inputs = self.v_processor(images=x_v, return_tensors="pt", padding=True)
            v_inputs = {k: v.to(self.device) for k, v in v_inputs.items()}
            t_inputs = self.t_processor(x_t, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_inputs = {k: v.to(self.device) for k, v in t_inputs.items()}
          
            t_outputs = self.t_processor(y_hat, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_outputs = {k: v.to(self.device) for k, v in t_outputs.items()}
            t_outputs_ref = self.t_processor(y, return_tensors="pt", padding=True, truncation=True, max_length=512)
            t_outputs_ref = {k: v.to(self.device) for k, v in t_outputs_ref.items()}
            
            attn_mask_i = t_inputs['attention_mask'].unsqueeze(-1)
            attn_mask_yh = t_outputs['attention_mask'].unsqueeze(-1)
            attn_mask_y = t_outputs_ref['attention_mask'].unsqueeze(-1)
            

            #pdb.set_trace()
            with torch.inference_mode():
                z_v = self.v_embedder.vision_model(pixel_values=v_inputs['pixel_values'], output_hidden_states=True).hidden_states[-1][:,1:,:].mean(dim=1).float()
                z_t = self.t_embedder(**t_inputs, output_hidden_states=True).hidden_states[0] * attn_mask_i
                z_yhat = self.t_embedder(**t_outputs, output_hidden_states=True).hidden_states[0] * attn_mask_yh
                z_y = self.t_embedder(**t_outputs_ref, output_hidden_states=True).hidden_states[0] * attn_mask_y
            
                z_t = z_t[:,1:,:].sum(dim=1) / attn_mask_i[:,1:,:].sum(dim=1)#.unsqueeze(-1)
                z_yhat = z_yhat[:,1:,:].sum(dim=1) / attn_mask_yh[:,1:,:].sum(dim=1)#.unsqueeze(-1)
                z_y = z_y[:,1:,:].sum(dim=1) / attn_mask_y[:,1:,:].sum(dim=1)#.unsqueeze(-1)


                z_v = F.normalize(z_v, p=2, dim=-1)
                z_t = F.normalize(z_t, p=2, dim=-1)
                z_yhat = F.normalize(z_yhat, p=2, dim=-1)
                z_y = F.normalize(z_y, p=2, dim=-1)

                z = (z_v+z_t)*0.5

                model_mi = self.mi_est(z, z_yhat).item()
                ref_mi = self.mi_est(z, z_y).item()
                emi = model_mi - ref_mi

            if return_emb:
                return emi, model_mi, ref_mi, z_v, z_t, z_yhat, z_y
        else:
            x = (x_v+x_t)/2
            x.to(self.device); y_hat.to(self.device); y.to(self.device)
            emi = self.mi_est(x, y_hat) - self.mi_est(x, y)

        return emi, model_mi, ref_mi

def EMIDupperbound(px_v, px_t, py_hat, py, qx_v, qx_t, qy_hat, qy, entropy_scaler=None):
    covPXv = torch.matmul(torch.t(px_v),px_v)
    covQXv = torch.matmul(torch.t(qx_v),qx_v)
    jsd_v = JSD_cov(covPXv,covQXv).item()

    covPXt = torch.matmul(torch.t(px_t),px_t)
    covQXt = torch.matmul(torch.t(qx_t),qx_t)
    jsd_t = JSD_cov(covPXt,covQXt).item()

    covPYH = torch.matmul(torch.t(py_hat),py_hat)
    covPY = torch.matmul(torch.t(py),py)
    jsd_py = JSD_cov(covPYH,covPY).item()

    covQYH = torch.matmul(torch.t(qy_hat),qy_hat)
    covQY = torch.matmul(torch.t(qy),qy)
    jsd_qy = JSD_cov(covQYH,covQY).item()
    
    if entropy_scaler is None:
        # scale-adjusted upper bound
        emid_ub = jsd_v**(1/2) + jsd_t**(1/2) + jsd_py**(1/4) + jsd_qy**(1/4)
        return emid_ub, jsd_v, jsd_t, jsd_py, jsd_qy
    else:
        # true upper bound estimate
        emid_ub = entropy_scaler*(jsd_v**(1/2) + jsd_t**(1/2)) + 4*(jsd_py**(1/4) + jsd_qy**(1/4))
        return emid_ub, jsd_v, jsd_t, jsd_py, jsd_qy