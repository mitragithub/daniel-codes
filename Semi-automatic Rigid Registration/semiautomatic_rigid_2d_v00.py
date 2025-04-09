

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set this before importing pyplot

import matplotlib.pyplot as plt
plt.set_loglevel('CRITICAL')

import torch
from os import makedirs
from os.path import split,join,splitext,basename
import tifffile

from scipy.interpolate import interpn
from scipy.ndimage import convolve,gaussian_filter

import argparse

fnameI =  '/home/dtward/mounts/mg5root/nfs/data/main/M38/mba_converted_imaging_data/MD963/MD963/MD963-N84-2023.08.30-00.21.33_MD963_2_0168.tif'
fnameJ = '/home/dtward/mounts/mg5root/nfs/data/main/M38/mba_converted_imaging_data/MD963/MD963/MD963-My84-2023.09.25-11.09.19_MD963_2_0168.tif'
dx = 0.46*32
gridsize = 3000.0
outputdir = 'twoDout'
manual = True
automatic = True
dtype = torch.float64
device = 'cuda:0'
ep = 0.2

parser = argparse.ArgumentParser(
                    prog='semiautomatic',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--fnameI',help='filename for the atlas/moving image (usually nissl)',required=True,type=str)
parser.add_argument('--fnameJ',help='filename for the target/fixed image (usually myelin)',required=True,type=str)
parser.add_argument('--outputdir',help='directory for outputs',required=True,type=str)
parser.add_argument('--manual',help='select points for initialization?',type=bool,default=True)
parser.add_argument('--automatic',help='run automatic registration?',type=bool,default=True)
parser.add_argument('--dx',help='pixel size',type=float,default=0.46*32)
parser.add_argument('--device',help='device (cpu, cuda:0, cuda:1, etc)',type=str,default='cpu')
parser.add_argument('--gridsize',help='size of overlay grid for QC figure',type=float,default=3000.0)
parser.add_argument('--blurry',help='do a first pass blurry registration',type=bool,default=False)

args = parser.parse_args()
print(args)
fnameI = args.fnameI
fnameJ = args.fnameJ
outputdir = args.outputdir
manual = args.manual
automatic = args.automatic
dx = args.dx
device = args.device
gridsize = gridsize
blurry = args.blurry







def rigid_alignment_block(xI, I, xJ, J, A=None, 
                    device='cuda:0', dtype=torch.float64, 
                    niter=2000, 
                    ep=1e-1, epL=1e-6, epT=1e-3, title='', ndraw=250):
    
    '''
    Rigid alignment with block matching.
    
    Every 100 iterations contrast coefficients are updated.
    
    Every 200 iterations, block size is reduced
    
    TODO
    ----
    Add voxel size.
    
    '''
    
    
    if A is None:
        A = torch.eye(3,requires_grad=True,device=device,dtype=dtype)
    else:
        A = torch.tensor(A,device=device,dtype=dtype,requires_grad=True)
    
    
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    I = torch.tensor(I,device=device,dtype=dtype)
    J = torch.tensor(J,device=device,dtype=dtype)
    XJ = torch.stack(torch.meshgrid(xJ,indexing='ij'),-1)
    XI = torch.stack(torch.meshgrid(xI,indexing='ij'),-1)
    dJ = [x[1] - x[0] for x in xI]
    DJ = torch.prod(torch.tensor(dJ,device=XI.device, dtype=XI.dtype))
    
    
    # let's create a metric
    dI = [x[1] - x[0] for x in xI]
    DI = torch.prod(torch.tensor(dI,device=XI.device,dtype=XI.dtype))
    g = torch.zeros(6,6,dtype=XI.dtype,device=XI.device)
    count = 0
    for i in range(2):
        for j in range(3):
            E = ((torch.arange(3)==i)[:,None] * (torch.arange(3)==j)[None,:]).to(dtype=XI.dtype,device=XI.device)
            EXI = (E[:2,:2]@XI[...,None])[...,0] + E[:2,-1]
            count_ = 0
            for i_ in range(2):
                for j_ in range(3):
                    E_ = ((torch.arange(3)==i_)[:,None] * (torch.arange(3)==j_)[None,:]).to(dtype=XI.dtype,device=XI.device)
                    E_XI = (E_[:2,:2]@XI[...,None])[...,0] + E_[:2,-1]

                    g[count,count_] = torch.sum(E_XI*EXI)*DI
                    count_ += 1
            count += 1


    
    
    
    extentI = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)
    extentJ = (xJ[1][0]-dJ[1]/2, xJ[1][-1]+dJ[1]/2, xJ[0][-1]+dJ[0]/2, xJ[0][0]-dJ[0]/2)
    
    extentI = ((xI[1][0]-dI[1]/2).item(), (xI[1][-1]+dI[1]/2).item(), (xI[0][-1]+dI[0]/2).item(), (xI[0][0]-dI[0]/2).item())
    extentJ = ((xJ[1][0]-dJ[1]/2).item(), (xJ[1][-1]+dJ[1]/2).item(), (xJ[0][-1]+dJ[0]/2).item(), (xJ[0][0]-dJ[0]/2).item())
    
    Esave = []
    fig,ax = plt.subplots(2,3)
    
    ax = ax.ravel()
    for it in range(niter):
        Ai = torch.linalg.inv(A)
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
        Xs_ = Xs.clone().detach()
        # scale 0 to 1
        Xs = Xs - torch.stack([xI[0][0],xI[1][0]])
        Xs = Xs / torch.stack([xI[0][-1] - xI[0][0],xI[1][-1] - xI[1][0]])
        # scale -1 to 1
        Xs = 2*Xs - 1
        # convert to xy
        Xs = Xs.flip(-1)
        # sample
        AI = torch.nn.functional.grid_sample(I[None],Xs[None],align_corners=True,padding_mode='border')[0]
        if False:
            # predict contrast
            B = torch.cat(   [torch.ones((1,J.shape[1]*J.shape[2]),device=device,dtype=dtype),AI.reshape(AI.shape[0],-1)] ,0)
            with torch.no_grad():    
                BB = B@B.T
                BJ = B@J.reshape(J.shape[0],-1).T
                coeffs = torch.linalg.solve(BB,BJ)
            fAI = (coeffs.T@B).reshape(J.shape)
        if True:
            '''
            # local contrast
            if it < 400:
                # start with the whole image
                M = torch.tensor(J.shape[1:],device=J.device)
            elif it == 800:
                M = M//2
                Mmax = torch.max(M)
                M = torch.tensor([Mmax,Mmax],device=J.device)
            elif it == 1200:
                M = M//2
            elif it == 1600:
                M = M//2
            elif it == 2000:
                M = M//2
            Mcut = 16
            if torch.any(M<Mcut):
                M = torch.tensor([Mcut,Mcut],device=J.device)
            '''    
            
            '''
            if it < 400:
                # start with the whole image
                M = torch.tensor(J.shape[1:],device=J.device)
            elif it == 800:
                M = torch.tensor([64,64],device=J.device)
            elif it == 1200:
                M = torch.tensor([32,32],device=J.device)
            elif it == 1600:
                M = torch.tensor([16,16],device=J.device)
            elif it == 2000:
                M = torch.tensor([8,8],device=J.device)
            '''
            
            '''
            if it < 500:
                # start with the whole image
                M = torch.tensor(J.shape[1:],device=J.device)
            elif it == 500:
                M = torch.tensor([128,128],device=J.device)
            elif it == 1000:
                M = torch.tensor([64,64],device=J.device)
            elif it == 1500:
                M = torch.tensor([32,32],device=J.device)
            elif it == 2000:
                M = torch.tensor([16,16],device=J.device)
            '''
            # now that I have a better start
            if it < 500:
                M = torch.tensor([64,64],device=J.device)
            elif it == 500:
                M = torch.tensor([32,32],device=J.device)
            elif it == 1000:
                M = torch.tensor([16,16],device=J.device)
                
            Jshape = torch.as_tensor(J.shape[1:],device=device)
            topad = Jshape%M
            topad = (M-topad)%M
            W = torch.ones_like(J[0][None])

            Jpad = torch.nn.functional.pad(J,(0,topad[1].item(),0,topad[0].item()))        
            AIpad = torch.nn.functional.pad(AI,(0,topad[1].item(),0,topad[0].item()))
            Wpad = torch.nn.functional.pad(W,(0,topad[1].item(),0,topad[0].item()))
            # now we will reshape it so that each block is a leading dimension
            #
            Jpad_ = Jpad.reshape( (Jpad.shape[0],Jpad.shape[1]//M[0].item(),M[0].item(),Jpad.shape[2]//M[1].item(),M[1].item()))
            Jpad__ = Jpad_.permute(1,3,2,4,0)
            Jpadv = Jpad__.reshape(Jpad__.shape[0],Jpad__.shape[1],(M[0]*M[1]).item(),Jpad__.shape[-1])

            AIpad_ = AIpad.reshape( (AIpad.shape[0],AIpad.shape[1]//M[0].item(),M[0].item(),AIpad.shape[2]//M[1].item(),M[1].item()))
            AIpad__ = AIpad_.permute(1,3,2,4,0)
            AIpadv = AIpad__.reshape(AIpad__.shape[0],AIpad__.shape[1],(M[0]*M[1]).item(),AIpad__.shape[-1],)

            Wpad_ = Wpad.reshape( (Wpad.shape[0],Wpad.shape[1]//M[0].item(),M[0].item(),Wpad.shape[2]//M[1].item(),M[1].item()))
            Wpad__ = Wpad_.permute(1,3,2,4,0)
            Wpadv = Wpad__.reshape(Wpad__.shape[0],Wpad__.shape[1],(M[0]*M[1]).item(),Wpad__.shape[-1],)

            # now basis function
            B = torch.cat((torch.ones_like(AIpadv[...,0])[...,None],AIpadv),-1)

            # coeffs
            ncoeffs = 10 # update every ncoeffs (was 100)
            if not it%ncoeffs:
                with torch.no_grad():            
                    BB = B.transpose(-1,-2)@(B*Wpadv)
                    BJ = B.transpose(-1,-2)@(Jpadv*Wpadv)
                    small = 1e-3
                    coeffs = torch.linalg.solve(BB + torch.eye(BB.shape[-1],device=BB.device)*small,BJ)
            fAIpadv = (B@coeffs).reshape(Jpadv.shape[0],Jpadv.shape[1],M[0].item(),M[1].item(),Jpadv.shape[-1])

            # reverse this permutation (1,3,2,4,0)
            fAIpad_ = fAIpadv.permute(4,0,2,1,3)        
            fAIpad = fAIpad_.reshape(Jpad.shape)

            fAI = fAIpad[:,:J.shape[1],:J.shape[2]]
        

            
        # note I changed sum to mean
        E = torch.sum((fAI - J)**2)/2.0*DI
        Esave.append(E.item())
        E.backward()

        # update
        with torch.no_grad():
            #A[:2,:2] -= A.grad[:2,:2]*epL
            #u,s,vh = torch.linalg.svd(A[:2,:2])
            #A[:2,:2] = u@vh
            #A[:2,-1] -= A.grad[:2,-1]*epT
            
            
            
            grad = A.grad[:2].reshape(-1)
            grad = torch.linalg.solve(g,grad)
            
            # we need stepsize estimation
            # to do this I need the transformed image, and the sample points A^{-1}x
            # transformed image is fAI, sample points is Xs_
            # 
            err = (fAI - J)            
            DfAI = torch.stack(torch.gradient(fAI,dim=(1,2),spacing=[d.item() for d in dJ]),-1)
            grad = grad.reshape(2,3)
            
            gradXs = (grad[:2,:2]@Xs_[...,None])[...,0] + grad[:2,-1]
            
            
            DfAIgradXs = (DfAI[...,None,:]@gradXs[None,...,None])[...,0,0]
            
            
            b = -2*torch.sum(err*DfAIgradXs)*DJ
            a = torch.sum(DfAIgradXs**2)*DJ
            step = b/a/2
            #print(step,ep)                        
            
            
            A[:2] -= ep*step*grad # do I have a sign error somewhere?
            u,s,vh = torch.linalg.svd(A[:2,:2])
            A[:2,:2] = u@vh
            
            A.grad.zero_()

        # draw
        if not it%ndraw or it == niter-1:
            with torch.no_grad():
                ax[0].cla()
                Ishow = AI.clone().detach().permute(1,2,0).cpu().numpy()
                if Ishow.shape[-1] > 3:
                    Ip,Is = pca(Ishow)
                    Is = pca_scale(Ip,Is)[...,:3]
                elif Ishow.shape[-1] == 3:
                    Is = Ishow
                elif Ishow.shape[-1] == 2:
                    Is = torch.stack((Ishow[...,0],Ishow[...,1],Ishow[...,0]),-1)
                elif Ishow.shape[-1] == 1:
                    Is = torch.stack((Ishow[...,0],Ishow[...,0],Ishow[...,0]),-1)
                ax[0].imshow(Is,extent=extentJ)
                ax[1].cla()
                ax[1].imshow(fAI.clone().detach().permute(1,2,0).cpu()[...,:3], extent=extentJ)
                if it == 0:
                    ax[2].cla()
                    ax[2].imshow(J.clone().detach().permute(1,2,0).cpu()[...,:3],extent=extentJ)
                ax[4].cla()
                ax[4].imshow(((fAI-J).clone().detach()*0.5*2+0.5).permute(1,2,0).cpu()[...,:3],extent=extentJ)
                ax[3].cla()
                ax[3].plot(Esave)
                fig.suptitle(title)
                fig.canvas.draw()

    fAI = torch.tensor(fAI)    
    return torch.tensor(A.clone().detach().cpu()),AI.clone().detach().cpu(),fAI.clone().detach().cpu(),fig



def blur(I):
    r = 5
    return gaussian_filter(I,(r,r,0))
    




print(f'Loading atlas image {fnameI}')
I = tifffile.imread(fnameI)
if I.dtype == np.uint8:
    I = I / 255.0
q = np.quantile(I,(0.01,0.99),axis=(0,1),)
I = (I - q[0,:])/(q[1,:]-q[0,:])
I = I.clip(0,1)    
print(f'Finished loading atlas image {fnameI}')

print(f'Loading target image {fnameJ}')
J = tifffile.imread(fnameJ)
if J.dtype == np.uint8:
    J = J / 255.0
q = np.quantile(J,(0.01,0.99),axis=(0,1),)
J = (J - q[0,:])/(q[1,:]-q[0,:])
J = J.clip(0,1)
#print(q)
print(f'Finished loading target image {fnameJ}')

print(f'Making output directory {outputdir}')
makedirs(outputdir,exist_ok=True)
print(f'Finished making output directory {outputdir}')
outnamebase = splitext(basename(fnameI))[0] + '_to_' + splitext(basename(fnameJ))[0] 
fulloutname_mat = join(outputdir,outnamebase+ '_xy_order.txt')
fulloutname_fig = join(outputdir,outnamebase+ '_QC.png')



nI = np.array(I.shape)
xI = [np.arange(n)*dx-(n-1)*dx/2 for n in nI[:2]]
extentI = (xI[-1][0]-dx/2,xI[-1][-1]+dx/2, xI[-2][-1]+dx/2, xI[-2][0]-dx/2)

nJ = np.array(J.shape)
xJ = [np.arange(n)*dx-(n-1)*dx/2 for n in nJ[:2]]
extentJ = (xJ[-1][0]-dx/2,xJ[-1][-1]+dx/2, xJ[-2][-1]+dx/2, xJ[-2][0]-dx/2)


fig,ax = plt.subplots(2,2,figsize=(10,5))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.9,wspace=0,hspace=0)
# fig.canvas.manager.window.showMaximized() #
fig.suptitle('Pick corresponding points on top left, then top right')
ax = ax.ravel()
ax[0].imshow(I,extent=extentI)
ax[0].set_title('atlas I')
ax[1].imshow(J,extent=extentJ)
ax[1].set_title('target J')
if manual:
    points = plt.ginput(-1,-1)
    #points = [(-3745.6398498805866, -4146.507485499831), (-1035.907653815535, -3275.1330376435762), (6397.047833503922, -2463.0738451040597), (8990.19856704196, -1314.307835778458), (1935.9486864551363, -3683.563234390993), (4624.587740247916, -2831.1726145797766), (-1220.4893892869331, 104.16245649948905), (1294.8845672694151, 1164.4711929944315), (4713.614193108155, 4607.3474445581705), (7177.36017286477, 5530.082019788471)]

    print('you selected points: ')
    print(points)
    points = np.array(points)
    # convert xy to rc
    points = points[:,::-1]
    qI = points[0::2]
    qJ = points[1::2]

    qIbar = np.mean(qI,0,keepdims=True)
    qJbar = np.mean(qJ,0,keepdims=True)

    qI0 = qI - qIbar
    qJ0 = qJ - qJbar

    S = qI0.T@qJ0
    u,s,vh = np.linalg.svd(S)

    R = (u@vh).T

    # so procedure
    # subtract qIbar
    # multiply by R
    # add qJbar
    T = qJbar - (R@qIbar[...,None])[...,0]


    #print(R,T)
    #print(R.shape,T.shape)
    # affine
    A = np.concatenate((R,T.T),1)
    A = np.concatenate((A,np.array([[0.0,0.0,1.0]])),0)
else:
    A = np.eye(3)
Ai = np.linalg.inv(A)
print('We computed initial guess')
print(A)


XJ = np.stack(np.meshgrid(*xJ,indexing='ij'),-1)
Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
AI = interpn(xI,I,Xs,bounds_error=False,fill_value=1)
grid0 = (xJ[0]%gridsize > gridsize/2)*2-1
grid1 = (xJ[1]%gridsize > gridsize/2)*2-1
grid = grid0[:,None]*grid1[None,:]*0.5+0.5
ax[2].imshow(AI*grid[...,None] + J*(1-grid)[...,None],extent=extentJ)
ax[2].set_title('initial alignment')

fig.canvas.draw()
plt.show(block=False)
plt.pause(0.001)


ndown = 2

Id = np.array(I)
xId = [np.array(x) for x in xI]
Jd = np.array(J)
xJd = [np.array(x) for x in xJ]

for i in range(ndown):
    print('downsampling by 2')
    print(Id.shape)
    ndI = np.array(Id.shape[:2])//2
    Id = Id[0:ndI[0]*2:2,0:ndI[1]*2:2]*0.25 + Id[1:ndI[0]*2:2,0:ndI[1]*2:2]*0.25 +  Id[0:ndI[0]*2:2,1:ndI[1]*2:2]*0.25 +  Id[1:ndI[0]*2:2,1:ndI[1]*2:2]*0.25
    xId = [xId[0][0:ndI[0]*2:2]*0.5+xId[0][1:ndI[0]*2:2]*0.5 ,xId[1][0:ndI[1]*2:2]*0.5+xId[1][1:ndI[1]*2:2]*0.5]

    # this downsampling is not working
    print(Jd.shape)
    ndJ = np.array(Jd.shape[:2])//2
    print(ndJ)
    Jd = Jd[0:ndJ[0]*2:2,0:ndJ[1]*2:2]*0.25 + Jd[1:ndJ[0]*2:2,0:ndJ[1]*2:2]*0.25 +  Jd[0:ndJ[0]*2:2,1:ndJ[1]*2:2]*0.25 +  Jd[1:ndJ[0]*2:2,1:ndJ[1]*2:2]*0.25
    xJd = [xJd[0][0:ndJ[0]*2:2]*0.5+xJd[0][1:ndJ[0]*2:2]*0.5, xJd[1][0:ndJ[1]*2:2]*0.5+xJd[1][1:ndJ[1]*2:2]*0.5]


if automatic:
    Ib = blur(Id)
    Jb = blur(Jd)

    print('Starting automatic registration')
    if blurry:
        A,AI,fAI,fig1 = rigid_alignment_block(xId,Ib.transpose(-1,0,1),xJd,Jb.transpose(-1,0,1), A=A, title='automatic registration',device=device,dtype=dtype,ep=ep)
        print(f'Finished blurry registration')


        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
        AI = interpn(xI,I,Xs,bounds_error=False,fill_value=1)
        grid0 = (xJ[0]%gridsize > gridsize/2)*2-1
        grid1 = (xJ[1]%gridsize > gridsize/2)*2-1
        grid = grid0[:,None]*grid1[None,:]*0.5+0.5
        ax[3].imshow(AI*grid[...,None] + J*(1-grid)[...,None],extent=extentJ)
        ax[3].set_title('first pass automatic')
        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.001)


    A,AI,fAI,fig1 = rigid_alignment_block(xId,Id.transpose(-1,0,1),xJd,Jd.transpose(-1,0,1), A=A, title='automatic registration',device=device,dtype=dtype,ep=ep)
    print('Done automatic registration')

    Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
    AI = interpn(xI,I,Xs,bounds_error=False,fill_value=1)
    grid0 = (xJ[0]%gridsize > gridsize/2)*2-1
    grid1 = (xJ[1]%gridsize > gridsize/2)*2-1
    grid = grid0[:,None]*grid1[None,:]*0.5+0.5
    ax[3].imshow(AI*grid[...,None] + J*(1-grid)[...,None],extent=extentJ)
    ax[3].set_title('final alignment')


# make outputs
# save fig1
print(A)
Axy = np.array(A)
Axy[[0,1]] = Axy[[1,0]]
Axy[:,[0,1]] = Axy[:,[1,0]]
print(Axy)
with open(fulloutname_mat,'wt') as f:
    f.write(f'{Axy[0,0]}, {Axy[0,1]}, {Axy[0,2]}\n')
    f.write(f'{Axy[1,0]}, {Axy[1,1]}, {Axy[1,2]}\n')
    f.write(f'{Axy[2,0]}, {Axy[2,1]}, {Axy[2,2]}\n')
fig.savefig(fulloutname_fig)



fig.canvas.draw()
plt.pause(0.001)
plt.show()


