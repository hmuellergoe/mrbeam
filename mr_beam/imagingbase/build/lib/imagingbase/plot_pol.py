#========================================================================================================================
#
# Load libraries--------------------------------------------------------------------------------------------------------- 
#
import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib        import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ehtim as eh
#

#
#========================================================================================================================
#
#                                                        Input Deck
#
#========================================================================================================================
#
# List the source/epoch/file name----------------------------------------------------------------------------------------
#
filename         = r'C:\Users\hendr\Documents\PhD\multiresolution_support\static_pol.fits'
img = eh.image.load_fits(filename)
cbar_lims = [0, 0.0025]

#
pcut = 0.1
nvec        =   10         # sets the density of the EVPA field (if = 1 the EVPA in every pixel will be displayed)
evpa_scalefactor =   100.0#1.0        # dimensionless multiplicative scale factor applied to *all* EVPAs
#

def display(img, axis, pcut=0.1, cbar_lims=[0, 1], nvec=10, evpa_scalefactor=1, has_cbar=True, has_label=True, has_title=True, power=1):
    plt.sca(axis)
    f = plt.gcf()
    
    #
    # Set the dimensions of the map------------------------------------------------------------------------------------------
    #
    xdim             =   img.xdim       # [pixels]
    ydim             =   img.ydim       # [pixels]
    #
    # Set the map scaling----------------------------------------------------------------------------------------------------
    #
    delt             =    img.psize*0.001/eh.RADPERUAS       # [mas per pixel]
    #
    # Set the plot limits----------------------------------------------------------------------------------------------------
    #
    plt_xmin         =    -img.fovx()/2*0.001/eh.RADPERUAS       # [mas]
    plt_xmax         =   img.fovx()/2*0.001/eh.RADPERUAS        # [mas]
    #
    plt_ymin         =   -img.fovy()/2*0.001/eh.RADPERUAS        # [mas]
    plt_ymax         =    img.fovy()/2*0.001/eh.RADPERUAS        # [mas]
    #
    #
    # Set the EVPA plotting parameters---------------------------------------------------------------------------------------
    evpa_cutoff      =   pcut*np.max(img.imvec)#0.035    # [Jy/beam] pixels with polarized intensity below this cutoff will be excluded from the EVPA field 
    # Set annotation parameters----------------------------------------------------------------------------------------------
    #
    # Set the axis font size
    #
    axis_fontsize      =  16        # [pt ]
    
    #========================================================================================================================
    #
    #                                               Perform Some Initial Calculations
    #
    #========================================================================================================================
    
    #
    # Setup Latex compatibility for plotting---------------------------------------------------------------------------------
    #
    rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
    rc('text',usetex=True)
    #
    #========================================================================================================================
    #
    #                                                     Read in the Data
    #
    #========================================================================================================================
    #
    # Load up the fits images------------------------------------------------------------------------------------------------
    #
    IMAP_data = np.asarray([[img.imvec.reshape((xdim, ydim))]])
    QMAP_data = np.asarray([[img.qvec.reshape((xdim, ydim))]])
    UMAP_data = np.asarray([[img.uvec.reshape((xdim, ydim))]])
    
    #
    #========================================================================================================================
    #
    #                                               Locate Peak in Total Intensity
    #
    #========================================================================================================================
    #
    # Locate the maximum of the total intensity------------------------------------------------------------------------------ 
    #
    IMAP_max = 0.0
    i_max = xdim//2
    j_max = ydim//2
    #
    IMAP_max = np.max(IMAP_data)
    #
    print("I-Max",IMAP_max)
    
    # Shift coordinates of the map so that the peak total intensity is located at the origin---------------------------------
    #
    xp = np.zeros( ( xdim, ydim ), float )
    yp = np.zeros( ( xdim, ydim ), float )
    
    #
    for i in range(0, xdim):
      xp[:,i] = ( i_max - i )*delt
    #
    for j in range(0, ydim):
      yp[j,:] = ( j - j_max )*delt
    #
    xmin = xp[0,0]
    xmax = xp[0,xdim-1]
    #
    ymin = yp[ydim-1,0]
    ymax = yp[0,0]
    
    #
    #========================================================================================================================
    #
    #                                                 Create plot
    #
    #========================================================================================================================
    #
    # Initialize the Map-----------------------------------------------------------------------------------------------------
    #
    # Create Polarization Intensity Array------------------------------------------------------------------------------------
    #
    PIMAP_data      = np.ndarray( shape=( ydim, xdim ), dtype=float )
    PIMAP_data[:,:] = np.sqrt( QMAP_data[0,0,:,:]**2 + UMAP_data[0,0,:,:]**2 )
    #
    # Locate the maximum of the polarized intensity--------------------------------------------------------------------------
    #
    PIMAP_max = 0.0
    #
    for i in range(0, xdim):
      for j in range(0, ydim):
    #
        if PIMAP_data[j,i] > PIMAP_max:
    #                 ^
    #     note: python arrays are inverted relative to Fortran (i & j indices are switched).
    #
          PIMAP_max = PIMAP_data[j,i]
    print("P-Max",PIMAP_max)
    #
    # Create the EVPA Array--------------------------------------------------------------------------------------------------
    #
    PAMMAP_data      = np.ndarray( shape=( ydim, xdim ), dtype=float )
    PAMMAP_data[:,:] = (( 0.5 )*( np.arctan2( UMAP_data[0,0,:,:], QMAP_data[0,0,:,:] )) )
    sumq=0
    sumu=0
    for i in range (0,xdim):
        for j in range(0,ydim):
            sumq=sumq+QMAP_data[0,0,i,j]
            sumu=sumu+UMAP_data[0,0,i,j]
    x= ( 0.5 )*( np.arctan2( sumu, sumq)  )
    print("EVPA", np.degrees(x))
    #
    # Overplot the Stokes I----------------------------------------------------------------------------------------------
    #
    axIm = plt.imshow(img.imarr()[::-1, ::-1], cmap='afmhot', extent = [ xmin, xmax, ymin, ymax ], vmin=cbar_lims[0], vmax=cbar_lims[1], interpolation='gaussian')
    # Overplot EVPAs---------------------------------------------------------------------------------------------------------
    #
    ax = axIm.axes
    thin = xdim // nvec
    #
    for j in range(0, ydim, thin):
      for i in range(0, xdim, thin):
        if IMAP_data[0, 0, j,i] > evpa_cutoff and np.isfinite( PAMMAP_data[j,i] ):
    #
          r  = (  PIMAP_data[j,i] )**power*( 0.5 )*( evpa_scalefactor )
          a  =  (( PAMMAP_data[j,i] ) + np.radians(2) )
    #
          x1 = xp[0,i] + r*np.sin(a)
          y1 = yp[j,0] + r*np.cos(a)
          x2 = xp[0,i] - r*np.sin(a)
          y2 = yp[j,0] - r*np.cos(a)
    #
    #      ax.add_line( lines.Line2D( ( -x1, -x2 ),( -y1, -y2 ), color=evpa_color, linewidth=evpa_thickness ) )
          #ax.quiver(-x1, -y1, x1-x2, y1-y2, headaxislength=20, headwidth=1, headlength=0.01, minlength=0, minshaft=1, pivot='mid', color='k', angles='uv', scale=1.0/thin)
          ax.quiver(-x1, -y1, -x1+x2, y1-y2,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1.0,
                   width=.01, pivot='mid', color='k', angles='uv', scale=1.0 / thin)
          ax.quiver(-x1, -y1, -x1+x2, y1-y2,
                   width=.005, headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   pivot='mid', color='w', angles='uv', scale=1.1 / thin)     
      
          
    # Label the axes---------------------------------------------------------------------------------------------------------
    #
    plt.xlim( plt_xmin, plt_xmax )
    plt.ylim( plt_ymin, plt_ymax )
    #
    
    if has_label:
        plt.xlabel( r"$\displaystyle \rm{Relative~RA } ~ [\rm{mas}]$", fontsize = 16 )
        plt.ylabel( r"$\displaystyle \rm{Relative~DEC} ~ [\rm{mas}]$", fontsize = 16 )
        #
        plt.xticks( fontsize = axis_fontsize )
        plt.yticks( fontsize = axis_fontsize )
    else:
        plt.xticks([])
        plt.yticks([])
    
    #
    # Setup the color bar----------------------------------------------------------------------------------------------------
    #
    if has_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="8%", pad=0.01) # par controls the position of the cbar
        cbar = plt.colorbar(axIm, orientation="horizontal", cax=cax)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.tick_params( labelsize = 12 )
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        #
        cbar.set_label( r"$\displaystyle \rm{ Intensity} ~ [\rm{Jy/pixel}]$", 
                         fontsize = 16, labelpad=-60, y=0.45 )
        #
        #cbar.ax.tick_params( labelsize = color_bar_fontsize )
        #cbar.formatter.set_powerlimits( (0, 0) )
        cbar.ax.xaxis.get_offset_text().set(size=1)
        #cbar.update_ticks()
        
    if has_title:
        plt.title("%s, m=%.1f percent" % (img.source, img.lin_polfrac() * 100), fontsize=16)
    
    return axis