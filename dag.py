'''
Function for Dense Adversarial Generation
Adversarial Examples for Semantic Segmentation
Muhammad Ferjad Naeem
ferjad.naeem@tum.de
adapted from https://github.com/IFL-CAMP/dense_adversarial_generation_pytorch
'''
import torch
from util import make_one_hot
import pdb

def DAG(model,image,ground_truth,adv_target,interp, num_iterations=20,gamma=0.07,no_background=True,background_class=0,device='cuda:0',verbose=False):
    '''
    Generates adversarial example for a given Image
    
    Parameters
    ----------
        model: Torch Model
        image: Torch tensor of dtype=float. Requires gradient. [b*c*h*w]
        ground_truth: Torch tensor of labels as one hot vector per class
        adv_target: Torch tensor of dtype=float. This is the purturbed labels. [b*classes*h*w]
        num_iterations: Number of iterations for the algorithm
        gamma: epsilon value. The maximum Change possible.
        no_background: If True, does not purturb the background class
        background_class: The index of the background class. Used to filter background
        device: Device to perform the computations on
        verbose: Bool. If true, prints the amount of change and the number of values changed in each iteration
    Returns
    -------
        Image:  Adversarial Output, logits of original image as torch tensor
        logits: Output of the Clean Image as torch tensor
        noise_total: List of total noise added per iteration as numpy array
        noise_iteration: List of noise added per iteration as numpy array
        prediction_iteration: List of prediction per iteration as numpy array
        image_iteration: List of image per iteration as numpy array
    '''
    

    noise_total=[]
    noise_iteration=[]
    prediction_iteration=[]
    image_iteration=[]
    background=None
    # pdb.set_trace()
    logits_feature5=model(image)[5]
    logits = interp(logits_feature5)
    orig_image=image
    _,predictions_orig=torch.max(logits,1)
    predictions_orig=make_one_hot(predictions_orig,logits.shape[1],device)

    # image.requires_grad_()
    
    pdb.set_trace()
    if(no_background):
        background=torch.zeros(logits.shape)
        background[:,background_class,:,:]=torch.ones((background.shape[2],background.shape[3]))
        background=background.to(device)
    
    for a in range(num_iterations):
        pdb.set_trace()
        output_feature5=model(image)[5]
        output = interp(output_feature5)
        _,predictions=torch.max(output,1)
        prediction_iteration.append(predictions[0].cpu().numpy())
        predictions=make_one_hot(predictions,logits.shape[1],device)

        pdb.set_trace()
        condition1=torch.eq(predictions,ground_truth)
        cond=condition1
       
        if no_background:
            condition2=(ground_truth!=background)
            cond=torch.mul(condition1,condition2)
        cond=cond.float()

        pdb.set_trace()
        if(cond.sum()==0):
            print("Condition Reached")
            image=None
            break
        
        #Finding pixels to purturb
        adv_log=torch.mul(output,adv_target)
        #Getting the values of the original output
        clean_log=torch.mul(output,ground_truth)

        pdb.set_trace()
        #Finding r_m
        adv_direction=adv_log-clean_log
        r_m=torch.mul(adv_direction,cond)
        r_m.requires_grad_()
        #Summation
        r_m_sum=r_m.sum()
        r_m_sum.requires_grad_()
        #Finding gradient with respect to image
        # r_m_grad=torch.autograd.grad(r_m_sum,image,retain_graph=True)
        r_m_grad=torch.autograd.grad(r_m_sum,image,retain_graph=True,allow_unused=True)
        #Saving gradient for calculation
        r_m_grad_calc=r_m_grad[0]
        
        #Calculating Magnitude of the gradient
        r_m_grad_mag=r_m_grad_calc.norm()
        
        pdb.set_trace()
        if(r_m_grad_mag==0):
            print("Condition Reached, no gradient")
            #image=None
            break
        #Calculating final value of r_m
        r_m_norm=(gamma/r_m_grad_mag)*r_m_grad_calc

        pdb.set_trace()
        #if no_background:
        #if False:
        if no_background is False:
            condition_image=cond.sum(dim=1)
            condition_image=condition_image.unsqueeze(1)
            r_m_norm=torch.mul(r_m_norm,condition_image)

        pdb.set_trace()
        #Updating the image
        #print("r_m_norm : ",torch.unique(r_m_norm))
        image=image + torch.clamp(r_m_norm,0,1)
        image_iteration.append(image[0][0].detach().cpu().numpy())
        noise_total.append((image-orig_image)[0][0].detach().cpu().numpy())
        noise_iteration.append(r_m_norm[0][0].cpu().numpy())

        pdb.set_trace()
        if verbose:
            print("Iteration ",a)
            print("Change to the image is ",r_m_norm.sum())
            print("Magnitude of grad is ",r_m_grad_mag)
            print("Condition 1 ",condition1.sum())
            if no_background:
                print("Condition 2 ",condition2.sum())
                print("Condition is", cond.sum()) 

    return image, logits, noise_total, noise_iteration, prediction_iteration, image_iteration