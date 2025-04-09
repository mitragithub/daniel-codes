'''
manual_nissl_3d_QC is used after automatic registration, to quality control and clean up 3D reconstruction of nissls.

Generally there are two things that need to be corrected.
1. Outlier slices
2. Wiggles from slice to slice

To correct one target nissl slice we display vtk files that were produced by the automatic pipeline in registered space
These images have already been rigidly aligned, but we may want to revise that alignment.
1. This target nissl slice  (the slice we want to adjust)
2. The left neighbor (if applicable)
3. The right neighbor (if applicable)
4. The MRI or 3D volume

Note, these images will all be saved as vtk files, and will all have the same number of pixels.

4 is used to make sure the 3D reconstruction is accurate (remove the banana effect)
2,3 are used to make sure it is not wiggly. They may be useful, when the nissl has a "nose/brainstem" but the MRI does not.
2,3,4 are called the neighbor images.
1 is called the target image.



The workflow is as follows.
1. make a copy of the registration outputs (only once, not once per slice)
2. Pick a target slice, I suggest the middle one
3. Run the code.

The code will replace the rigid transformation matrix with a new one.
AND it will replace the nissl image vtk file with a new one.
** Note this last step will lead to double interpolation, so it is not ideal, but simplifies the interface
** these low res vtk files are really only used for QC so it is okay

4. Move on to the next slice
5. Potentially two people can be working together, one starts at the middle and goes left.
One starts to the right of the middle and goes right.
Only one person does each slice
All the slices are done in order, so we can look at the neighbor that you have already corrected.

'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from os.path import basename, join, split, splitext
from emlddmm import emlddmm
from os import makedirs



def main(args):
    verbose = args.verbose
    if verbose: print(f'Reading target image {args.target_nissl_file}')
    xI0 = None
    if args.target_nissl_file.endswith('.vtk'):
        print('ends with vtk')        
        fname = args.target_nissl_file                    
        xI,I,titleI,namesI = emlddmm.read_data(fname)
        xI0 = xI.pop(0)
        dI = [x[1] - x[0] for x in xI]
        # move color channel last?  and convert 3D to 2D
        I = I.transpose(1,2,3,0).squeeze()
        nI = np.array(I.shape)
        print(nI)
    else:
        raise Exception('Only VTK files are supported')
    extentI = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0])
        

    args.neighbor_image_files = [args.neighbor_nissl_file, args.neighbor_mri_file]
    if verbose: print(f'Reading neighbor images {args.neighbor_image_files}')
    nneighbors = len(args.neighbor_image_files)
    xJ0 = []
    xJ = []
    nJ = []
    dJ = []
    extentJ = []
    J = []
    for i in range(nneighbors):
        if args.neighbor_image_files[i].endswith('.vtk'):
            print('ends with vtk')            
            xJ_,J_,titleJ,namesJ = emlddmm.read_data(args.neighbor_image_files[i]) 
            try:  
                dJ_ = [x[1] - x[0] for x in xJ_]
            except:
                print('Warning, inserting 20 for slice width')
                dJ_ = [20.0,  xJ_[1][1] - xJ_[1][0], xJ_[2][1] - xJ_[2][0]]

            # move color channel last?  and convert 3D to 2D
            if J_.shape[0] == 1:
                J_ = np.repeat(J_,3,axis=0)
            J_ = J_.transpose(1,2,3,0).squeeze() # squeeze out the space dimension
            
            nJ_ = np.array(J_.shape)
            xJ0.append(xJ_.pop(0))
            xJ.append(xJ_)
            nJ.append(nJ_)
            dJ.append(dJ_)
            J.append(J_)
            print(nJ_)
        else:
            raise  Exception('only VTK supported')            
        extentJ.append(  (xJ_[1][0]-dJ_[1]/2, xJ_[1][-1]+dJ_[1]/2, xJ_[0][-1]+dJ_[0]/2, xJ_[0][0]-dJ_[0]) )

    if args.normalization is None:
        if verbose: print('not normalizing')    
    else:
        if verbose: print(f'normalizing by {args.normalization} percentile')
        for i in range(nneighbors):
            J[i] = J[i] / np.quantile(J[i],float(args.normalization)/100.0,axis=(0,1),keepdims=True)
        I = I / np.quantile(I,float(args.normalization)/100.0,axis=(0,1),keepdims=True)
        
    # in registered space all sampling grids are the same
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),-1)
    XJ = np.stack(np.meshgrid(*xI,indexing='ij'),-1)
        
    AI = I
    AJ = J

    
    fig,ax = plt.subplots(2,2)
    ax = [ax[0,0],ax[0,1],ax[1,0]]    
    h_imageI = ax[0].imshow(AI,extent=extentI) # they will all be in extent J now
    ax[-1].set_title('Target')
    
    h_imageJ = []
    for i in range(nneighbors):
        h_imageJ.append( ax[i].imshow(AJ[i],extent=extentI) )
        if i == 0:
            ax[i].set_title('neighbor nissl')
        elif i == 1:
            ax[i].set_title('neighbor mri')


    if args.grid is None:
        distance_0 = np.abs(xI[0][-1] - xI[0][0])
        distance_1 = np.abs(xI[1][-1] - xI[1][0])
        distance = np.min([distance_0, distance_1])
        grid0 = grid1 = distance/5.0
    else:
        grid0 = grid1 = args.grid
    
    squares0 = xI[0]%grid0 >= grid0/2
    squares1 = xI[1]%grid1 >= grid1/2
    Squares = ((squares0[:,None]-0.5)*(squares1[None,:]-0.5) ) > 0
    
    Squares = Squares[...,None]
    print(AI.shape,AJ[0].shape,Squares.shape)
    #h_imageIJ = ax[2].imshow(AI*Squares + AJ[0]*(1.0 - Squares),extent=extentI)
    #ax[2].set_title('Transformed with overlay')
    h_imageI = ax[-1].imshow(AI,extent=extentI)
    ax[-1].set_title('Target image')

    pointsTN = []
    pointsNN = []
    pointsNM = []
    h_pointsTN = None
    h_pointsNN = None
    h_pointsNM = None    
    
    ANN = np.eye(3)
    ANM = np.eye(3)
    while True:
                   
        fig.suptitle('Click a point on target nissl image (enter to finish)')        
        plt.pause(0.001)
        pointTN = fig.ginput(timeout=0)
        if not pointTN: break
        pointsTN.extend(pointTN)
        pointsTN_ = np.array(pointsTN)[:,::-1] # row column order
        
        if h_pointsTN is not None:
            h_pointsTN.remove()
        h_pointsTN = ax[-1].scatter(pointsTN_[:,1],pointsTN_[:,0],c='b')
        if verbose: print(f'point Nissl Target {pointTN}')






        fig.suptitle('Click a point on neighbor nissl image (enter to finish)')
        plt.pause(0.001)
        pointNN = fig.ginput(timeout=0)
        if not pointNN: break
        pointsNN.extend(pointNN)
        pointsNN_ = np.array(pointsNN)[:,::-1] # row column order
        
        if h_pointsNN is not None:
            h_pointsNN.remove()

        h_pointsNN = ax[0].scatter(pointsNN_[:,1],pointsNN_[:,0],c='r')                
        if verbose: print(f'point NN {pointNN}')

        


        fig.suptitle('Click a point on neighbor mri image (enter to finish). if no mri is visible just press enter')
        plt.pause(0.001)
        pointNM = fig.ginput(timeout=0)
        if not pointNM: 
            pass
        else:
            pointsNM.extend(pointNM)
            pointsNM_ = np.array(pointsNM)[:,::-1] # row column order
            
            if h_pointsNM is not None:
                h_pointsNM.remove()

            h_pointsNM = ax[1].scatter(pointsNM_[:,1],pointsNM_[:,0],c='r')                
            if verbose: print(f'point NM {pointNM}')

        

    
        # now find the optimal transform
        if len(pointsTN)<2:
            ANN = np.eye(3)
            ANN[:2,-1] = np.mean(pointsNN_,0) - np.mean(pointsTN_,0)
            if len(pointsNM):
                ANM = np.eye(3)
                ANM[:2,-1] = np.mean(pointsNM_,0) - np.mean(pointsTN_,0)
        else:
            # first subtract com
            comTN = np.mean(pointsTN_,0)
            comNN = np.mean(pointsNN_,0)
            pointsTN0 = pointsTN_ - comTN
            pointsNN0 = pointsNN_ - comNN
            # now find the cross covariance
            S = pointsTN0.T@pointsNN0
            # now find the svd
            u,s,vh = np.linalg.svd(S)
            # now the rotation
            R = vh.T@u
            # now the translation
            T = comNN - R@comTN
            ANN = np.eye(3)
            ANN[:2,:2] = R
            ANN[:2,-1] = T            

            if len(pointsNM):
                # first subtract com
                comTN = np.mean(pointsTN_,0)
                comNM = np.mean(pointsNM_,0)
                pointsTN0 = pointsTN_ - comTN
                pointsNM0 = pointsNM_ - comNM
                # now find the cross covariance
                S = pointsTN0.T@pointsNM0
                # now find the svd
                u,s,vh = np.linalg.svd(S)
                # now the rotation
                R = vh.T@u
                # now the translation
                T = comNM - R@comTN
                ANM = np.eye(3)
                ANM[:2,:2] = R
                ANM[:2,-1] = T        
        #pointsIJ_ = (A[:2,:2]@pointsI_.T).T + A[:2,-1]
        ANNi = np.linalg.inv(ANN)
        if len(pointsNM):
            ANMi = np.linalg.inv(ANM)
        
       
    Xs = (ANNi[:2,:2]@XJ[...,None])[...,0] + ANNi[:2,-1]   
    ANNI = interpn(xI,AI,Xs,bounds_error=False,fill_value=0) # note both images are sampled on the pixels in J    
    h_imageIJ = ax[0].imshow(ANNI*Squares + AJ[0]*(1.0 - Squares),extent=extentI)

    if len(pointsNM):
        Xs = (ANMi[:2,:2]@XJ[...,None])[...,0] + ANMi[:2,-1]    
        ANMI = interpn(xI,AI,Xs,bounds_error=False,fill_value=0) # note both images are sampled on the pixels in J    
        h_imageIJ = ax[1].imshow(ANMI*Squares + AJ[1]*(1.0 - Squares),extent=extentI)
    plt.pause(0.001)
    plt.show()

    # write out the points
    makedirs(args.output_directory,exist_ok=True)
    output = str(pointsTN_) + str(pointsNN_)
    if len(pointsNM):
        output = output + str(pointsNM_)
    print(output)
    outname = join(args.output_directory, basename(args.target_nissl_file).replace('.vtk','_output_points.txt'))
    print(outname)
    with open(outname,'wt') as f:       
        print('file opened for writing') 
        f.write(output)


    
    
if __name__ == '__main__':
    print('hello world')

    # set up the parse
    parser = argparse.ArgumentParser(
        prog='python manual_nissl_3d_QC.py',
        description='This program will produce rigid transforms from mouse clicks to adjust a 3',
        epilog='Author: Daniel Tward'
    )
    # add arguments        
    parser.add_argument('-T','--target-nissl-file',type=str,required=True,help='VTK filename for the target nissl image in registered space, i.e. the Nissl slice you want to adjust).  This will be adjusted and replaced.')    
    parser.add_argument('-N','--neighbor-nissl-file',type=str,required=True,help='Filename for the neighbor nissl images in registered space')
    parser.add_argument('-M','--neighbor-mri-file',type=str,required=True,help='Filename for the neighbor images in registered space')
    parser.add_argument('-O','--output-directory',type=str,required=True,help='Directory for outputs')
    
    #parser.add_argument('-o','--transform-output',type=str,help='Name of output file (default: {moving}_to_{fixed}_matrix.txt)')
    #parser.add_argument('-i','--inverse-output',type=str,help='Name of output file for inverse (default: {fixed}_to_{moving}_matrix.txt)')
    #parser.add_argument('-t','--transformation',type=str,choices=['translation','rigid','affine'],default='rigid',help='Transformation type (default rigid)')
    #parser.add_argument('--moving-origin-row',type=float,default=None,help='Coordinate of the first row of the moving image (default: origin is in center of image)')
    #parser.add_argument('--moving-origin-col',type=float,default=None,help='Coordinate of the first column of the moving image (default: origin is in center of image)')
    #parser.add_argument('--moving-resolution-row',type=float,default=0.46*32,help='Resolution of the rows of the moving image (default: 0.46*32)')
    #parser.add_argument('--moving-resolution-col',type=float,default=0.46*32,help='Resolution of the columns of the moving image (default: 0.46*32)')
    #parser.add_argument('--fixed-origin-row',type=float,default=None,help='Coordinate of the first row of the fixed image (default: origin is in center of image)')
    #parser.add_argument('--fixed-origin-col',type=float,default=None,help='Coordinate of the first column of the fixed image (default: origin is in center of image)')
    #parser.add_argument('--fixed-resolution-row',type=float,default=0.46*32,help='Resolution of the rows of the fixed image (default: 0.46*32)')
    #parser.add_argument('--fixed-resolution-col',type=float,default=0.46*32,help='Resolution of the columns of the fixed image (default: 0.46*32)')
    parser.add_argument('-v','--verbose',type=bool,default=False,help='Print info useful for debugging')
    #parser.add_argument('-l','--layout',type=int,default=1,choices=[1,2,3],help='1 for 2x2 layout, 2 for 3x2 layout, 3 for 2x3 layout',)
    parser.add_argument('-g','--grid',type=float,default=None,help='Size of grid in overlay (default is one fifth of smallest dimension)')
    parser.add_argument('-n','--normalization',type=float, default=99, help='Normalize images for display if not none (default).  Enter a number from 0 to 100 to normalize by this percentile.  This is often necessary for high dynamic range images (like fluorescence).')    
    #parser.add_argument('--moving-initial-affine',type=str,default=None, help='an initial affine transform for the moving image (text file).  This is expected to be a "registered_to_input" matrix, which will be used to reconstruct the image in registered space and would be located in the registered space folder.')
    #parser.add_argument('--fixed-initial-affine',type=str,default=None, help='an initial affine transform for the fixed image(text file). This is expected to be a "registered_to_input" matrix, which will be used to reconstruct the image in registered space and would be located in the registered space folder.')
    #parser.add_argument('--moving-output-file',type=str,default=None, help='A filename for the transformed moving image.')
    

    # parse
    args = parser.parse_args()
    print(args)

    main(args)
