'''
manual_nissl_3d_QC is used after automatic registration, to quality control and clean up 3D reconstruction of nissls.

Generally there are two things that need to be corrected.
1. Outlier slices
2. Wiggles from slice to slice

To correct one target nissl slice we display vtk files
1. This target nissl slice 
2. The left neighbor (if applicable)
3. The right neighbor (if applicable)
4. The MRI or 3D volume

4 is used to make sure the 3D reconstruction is accurate
2,3 are used to make sure it is not wiggly.
2,3,4 are called the neighbor images.



The workflow is as follows.
1. make a copy of the registration outputs (only once, not once per slice)
2. Pick a target slice, I suggest the middle one
3. Run the code.

The code will replace the transformation matrix with a new one.
AND it will replace the nissl vtk file with a new one.
** Note this last step will lead to double interpolation, so it is not ideal, but simplifies the interface
** these low res vtk files are really only used for QC so it is okay


'''

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from os.path import basename, join, split, splitext
from emlddmm import emlddmm
import sys



def main(args):
    verbose = args.verbose
    if verbose: print(f'Reading target image {args.target_image_file}')
    xI0 = None
    if args.target_image_file.endswith('.vtk'):
        print('ends with vtk')        
        if args.target_image_file_orig is not None:
            fname = args.target_image_file_orig
        else:
            fname = args.target_image_file
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
    
    # let's load any existing matrices
    
    # read in xy order
    if args.target_rigid_file_orig is not None:
        fname = args.target_rigid_file_orig
    else:
        fname = args.target_rigid_file
    with open(fname) as f:
        A0 = []
        for line in f:                
            A0.append([float(x) for x in line.split(',')])
        A0 = np.array(A0)
        
        A0 = A0[[1,0,2]]
        A0 = A0[:,[1,0,2]]
    AI = I
    AJ = J

    
    fig,ax = plt.subplots(2,3)
    ax = [ax[0,0],ax[0,1],ax[0,2],ax[1,1]]    
    h_imageI = ax[0].imshow(AI,extent=extentI) # they will all be in extent J now
    ax[-1].set_title('Target')
    
    h_imageJ = []
    for i in range(nneighbors):
        h_imageJ.append( ax[i].imshow(AJ[i],extent=extentI) )
        ax[i].set_title('neighbor')


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

    pointsI = []
    pointsJ = []
    h_pointsI = None
    h_pointsJ = None
    h_pointsIJ = None
    h_pointsJJ = None
    A = np.eye(3)
    
    def on_key(event):
        if event.key == 'a':  # 'a' for abort
            plt.close('all')
            raise KeyboardInterrupt("Processing aborted by user")

    # Connect the key event to all axes
    for a in ax:
        a.figure.canvas.mpl_connect('key_press_event', on_key)

    while True:
        try:
            fig.suptitle('Click points on target image (enter to finish, press "a" to abort)')        
            plt.pause(0.001)
            pointI = fig.ginput(timeout=0)
            if not pointI: break
            pointsI.extend(pointI)
            pointsI_ = np.array(pointsI)[:,::-1] # row column order
            
            if h_pointsI is not None:
                h_pointsI.remove()
            h_pointsI = ax[-1].scatter(pointsI_[:,1],pointsI_[:,0],c='b')
            if verbose: print(f'point I {pointI}')

            fig.suptitle('Click points on neighbor image (enter to finish, press "a" to abort)')
            plt.pause(0.001)
            pointJ = fig.ginput(timeout=0)
            if not pointJ: break
            pointsJ.extend(pointJ)
            pointsJ_ = np.array(pointsJ)[:,::-1] # row column order
            
            if h_pointsJ is not None:
                h_pointsJ.remove()
            h_pointsJ0 = ax[0].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
            h_pointsJ1 = ax[1].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
            h_pointsJ2 = ax[2].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
            if verbose: print(f'point J {pointJ}')

            # now find the optimal transform
            if len(pointsI)<2:
                A = np.eye(3)
                A[:2,-1] = np.mean(pointsJ_,0) - np.mean(pointsI_,0)
            else:
                # first subtract com
                comI = np.mean(pointsI_,0)
                comJ = np.mean(pointsJ_,0)
                pointsI0 = pointsI_ - comI
                pointsJ0 = pointsJ_ - comJ
                # now find the cross covariance
                S = pointsI0.T@pointsJ0
                # now find the svd
                u,s,vh = np.linalg.svd(S)
                # now the rotation
                R = vh.T@u
                # now the translation
                T = comJ - R@comI            
                A = np.eye(3)
                A[:2,:2] = R
                A[:2,-1] = T            

            pointsIJ_ = (A[:2,:2]@pointsI_.T).T + A[:2,-1]
            Ai = np.linalg.inv(A)
            
            '''
            Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
            print(f'transforming moving image using matrix')
            print(Ai)
            AAI = interpn(xI,AI,Xs,bounds_error=False,fill_value=0) # note both images are sampled on the pixels in J
            if h_imageIJ is not None:
                h_imageIJ.remove()
            h_imageIJ = ax[2].imshow(AAI*Squares + AJ*(1.0 - Squares),extent=extentJ)
            if h_pointsIJ is not None:
                h_pointsIJ.remove()
            h_pointsIJ = ax[2].scatter(pointsIJ_[:,1],pointsIJ_[:,0],fc='none',ec='m')
            if h_pointsJJ is not None:
                h_pointsJJ.remove()
            h_pointsJJ = ax[2].scatter(pointsJ_[:,1],pointsJ_[:,0],c='r')
            plt.pause(0.001)
            '''
        except KeyboardInterrupt:
            print("Processing aborted by user")
            sys.exit(1)  # Exit with error code

    Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]
    print(f'transforming moving image using matrix')
    print(Ai)
    print(AI.shape,Xs.shape)    
    AAI = interpn(xI,AI,Xs,bounds_error=False,fill_value=0) # note both images are sampled on the pixels in J
    print(AAI.shape)
    for i in range(3):
        # Add debug print to check lengths
        print(f"Length of AJ: {len(AJ)}, Current i: {i}")
        
        # Normalize image data to [0,1] range before display
        AAI = (AAI - np.min(AAI)) / (np.max(AAI) - np.min(AAI))
        
        # Ensure i is within bounds of AJ
        if i < len(AJ):
            AJ[i] = (AJ[i] - np.min(AJ[i])) / (np.max(AJ[i]) - np.min(AJ[i]))
            h_imageIJ = ax[i].imshow(AAI*Squares + AJ[i]*(1.0 - Squares), extent=extentI)
        else:
            print(f"Warning: Index {i} is out of range for AJ (length {len(AJ)})")
            h_imageIJ = ax[i].imshow(AAI, extent=extentI)
    plt.pause(0.001)
    plt.show()

    # now we write out    

    # okay now what do I actually want to write out?
    # some combination of A0 and A
    # the sequence of transforms I apply to my image sample points
    # 1. Apply A0
    # 2. Apply A^{-1}
    # that means the sequence of operations I would apply to points in the space is
    # 1. Apply A
    # 2. Apply A0^{-1}
    # we hope that the first one is a better version of A0
    # we hope the second one is a better version of A0^{-1}
    # note that the second one is my "forward matrix"
    # the first one is my "inverse matrix"
    # as a double check, if I use my "inverse matrix" as an initial guess for moving, it should appear aligned to fixed
    print(A)
    #A_ = np.linalg.inv(A0)@A # this looked incorrect
    A_ = A@np.linalg.inv(A0) # try this, this looked correct
    print('writing out updated matrix')
    print(A_)
    Aout = A_.copy()[[1,0,2]] # swap xy
    Aout = Aout[:,[1,0,2]]    # swap xy
    print(Aout)
    with open(args.target_rigid_file,'wt') as f:
        f.write(f'{Aout[0,0]}, {Aout[0,1]}, {Aout[0,2]}\n')
        f.write(f'{Aout[1,0]}, {Aout[1,1]}, {Aout[1,2]}\n')
        f.write(f'{Aout[2,0]}, {Aout[2,1]}, {Aout[2,2]}\n')
        

    # output transformed images
    
    # this one is already calculated
    # it's called AAI
    # write it out here
    # but on what sampling grid? The sampling grid of the fixed image.  For our nose correct, these are the same. In other cases I may want something else.
    # if it is supposed to be some kind of a "fix" maybe it should be on its original sampling grid
    # I should check by regenerating AAI wit A_                        
    
    if xI0 is not None:
        print(f'Setting origin to 0 and and slice thickness to 20 for vtk output')
        xJuse = [[0.0,20.0],xI[0],xI[1]]
    else:
        print(f'Setting slice thickness to 20 for vtk output')
        xJuse = [[xI0,xI0+20.0],xI[0],xI[1]]
    emlddmm.write_data(args.target_image_file, xJuse, AAI[None].transpose(-1,0,1,2), 'manually edited' )
        
    
    
    
    
if __name__ == '__main__':
    print('hello world')

    # set up the parse
    parser = argparse.ArgumentParser(
        prog='python manual_nissl_3d_QC.py',
        description='This program will produce rigid transforms from mouse clicks to adjust a 3',
        epilog='Author: Daniel Tward'
    )
    # add arguments        
    parser.add_argument('-T','--target-image-file',type=str,required=True,help='VTK filename for the target image in registered space, i.e. the Nissl slice you want to adjust).  This will be adjusted and replaced.')    
    parser.add_argument('-N','--neighbor-image-files',type=str,required=True,help='Filename for the neighbor images in registered space', action='append')
    parser.add_argument('-R','--target-rigid-file',type=str,required=True,help='Current rigid transformation matrix which will be adjusted and replaced.  Note this transform must match the target image.')
    parser.add_argument('--target-image-file-orig',type=str,required=False,help='VTK filename for the target image in registered space, i.e. the Nissl slice you want to adjust). This will be adjusted but NOT replaced.  Useful for restoring from the original dataset.')
    parser.add_argument('--target-rigid-file-orig',type=str,required=False,help='Current rigid transformation which will be adjusted but NOT replaced.  Useful for restoring from the original datset. Note this transform must match the target image.')
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