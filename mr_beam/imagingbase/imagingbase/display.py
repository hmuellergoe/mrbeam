#This function is cherry-picked from the ehtim repository, ehtim.image.display with only slight modifications

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh

def display(img, pol=None, cfun=False, interp='gaussian',
            scale='lin', gamma=0.5, dynamic_range=1.e3,
            plotp=False, plot_stokes=False, nvec=20,
            vec_cfun=None,
            pcut=0.1, mcut=0.01, log_offset=False,
            label_type='ticks', has_title=True, alpha=1,
            has_cbar=True, only_cbar=False, cbar_lims=(), cbar_unit=('Jy', 'pixel'),
            export_pdf="", pdf_pad_inches=0.0, show=True, beamparams=None,
            cbar_orientation="vertical", scinot=False,
            scale_lw=1, beam_lw=1, cbar_fontsize=12, axis=None,
            scale_fontsize=12, power=0, beamcolor='w', dpi=500, vec_cfun_min=0, vec_cfun_max=1):
    """Display the image.
    
       Args:
           pol (str): which polarization image to plot. Default is img.pol_prim
                      pol='spec' will plot spectral index!
           cfun (str): matplotlib.pyplot color function.
                       False changes with 'pol',  but is 'afmhot' for most
           interp (str): image interpolation 'gauss' or 'lin'
    
           scale (str): image scaling in ['log','gamma','lin']
           gamma (float): index for gamma scaling
           dynamic_range (float): dynamic range for log and gamma scaling
    
           plotp (bool): True to plot linear polarimetic image
           plot_stokes (bool): True to plot stokes subplots along with plotp
           nvec (int): number of polarimetric vectors to plot
           vec_cfun (str): color function for vectors colored by |m|
    
           pcut (float): minimum stokes I value for displaying polarimetric vectors
                         (fraction of maximum Stokes I)
           mcut (float): minimum fractional polarization value for displaying vectors
           label_type (string): specifies the type of axes labeling: 'ticks', 'scale', 'none'
           has_title (bool): True if you want a title on the plot
           has_cbar (bool): True if you want a colorbar on the plot
           cbar_lims (tuple): specify the lower and upper limit of the colorbar
           cbar_unit (tuple): specifies the unit of the colorbar: e.g.,
                              ('Jy','pixel'),('m-Jy','$\mu$as$^2$'),['Tb']
           beamparams (list): [fwhm_maj, fwhm_min, theta], set to plot beam contour
    
           export_pdf (str): path to exported PDF with plot
           show (bool): Display the plot if true
           scinot (bool): Display numbers/units in scientific notation
           scale_lw (float): Linewidth of the scale overlay
           beam_lw (float): Linewidth of the beam overlay
           cbar_fontsize (float): Fontsize of the text elements of the colorbar
           axis (matplotlib.axes.Axes): An axis object
           scale_fontsize (float): Fontsize of the scale label
           power (float): Passed to colorbar for division of ticks by 1e(power)
           beamcolor (str): color of the beam overlay
    
       Returns:
           (matplotlib.figure.Figure): figure object with image
    
    """
    
    if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
        interp = 'gaussian'
    else:
        interp = 'linear'
    
    if not(beamparams is None or beamparams is False):
        if beamparams[0] > img.fovx() or beamparams[1] > img.fovx():
            raise Exception("beam FWHM must be smaller than fov!")
    
    if img.polrep == 'stokes' and pol is None:
        pol = 'I'
    elif img.polrep == 'circ' and pol is None:
        pol = 'RR'
    
    if only_cbar:
        has_cbar = True
        label_type = 'none'
        has_title = False
    
    if axis is None:
        f = plt.figure()
        plt.clf()
    
    if axis is not None:
        plt.sca(axis)
        f = plt.gcf()
    
    # Get unit scale factor
    factor = 1.
    fluxunit = 'Jy'
    areaunit = 'pixel'
    
    if cbar_unit[0] in ['m-Jy', 'mJy']:
        fluxunit = 'mJy'
        factor *= 1.e3
    elif cbar_unit[0] in ['muJy', r'$\mu$-Jy', r'$\mu$Jy']:
        fluxunit = r'$\mu$Jy'
        factor *= 1.e6
    elif cbar_unit[0] == 'Tb':
        factor = 3.254e13 / (img.rf**2 * img.psize**2)
        fluxunit = 'Brightness Temperature (K)'
        areaunit = ''
        if power != 0:
            fluxunit = (r'Brightness Temperature ($10^{{' + str(power) + '}}$ K)')
        else:
            fluxunit = 'Brightness Temperature (K)'
    elif cbar_unit[0] in ['Jy']:
        fluxunit = 'Jy'
        factor *= 1.
    else:
        factor = 1
        fluxunit = cbar_unit[0]
        areaunit = ''
    
    if len(cbar_unit) == 1 or cbar_unit[0] == 'Tb':
        factor *= 1.
    
    elif cbar_unit[1] == 'pixel':
        factor *= 1.
        if power != 0:
            areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')
    
    elif cbar_unit[1] in ['$arcseconds$^2$', 'as$^2$', 'as2']:
        areaunit = 'as$^2$'
        fovfactor = img.xdim * img.psize * (1 / ehc.RADPERAS)
        factor *= (1. / fovfactor)**2 / (1. / img.xdim)**2
        if power != 0:
            areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')
    
    elif cbar_unit[1] in [r'$\m-arcseconds$^2$', 'mas$^2$', 'mas2']:
        areaunit = 'mas$^2$'
        fovfactor = img.xdim * img.psize * (1 / ehc.RADPERUAS) / 1000.
        factor *= (1. / fovfactor)**2 / (1. / img.xdim)**2
        if power != 0:
            areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')
    
    elif cbar_unit[1] in [r'$\mu$-arcseconds$^2$', r'$\mu$as$^2$', 'muas2']:
        areaunit = r'$\mu$as$^2$'
        fovfactor = img.xdim * img.psize * (1 / ehc.RADPERUAS)
        factor *= (1. / fovfactor)**2 / (1. / img.xdim)**2
        if power != 0:
            areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')
    
    elif cbar_unit[1] == 'beam':
        if (beamparams is None or beamparams is False):
            print("Cannot convert to Jy/beam without beamparams!")
        else:
            areaunit = 'beam'
            beamarea = (2.0 * np.pi * beamparams[0] * beamparams[1] / (8.0 * np.log(2)))
            factor = beamarea / (img.psize**2)
            if power != 0:
                areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')
    
    else:
        raise ValueError('cbar_unit ' + cbar_unit[1] + ' is not a possible option')
    
    if not plotp:  # Plot a single polarization image
        cbar_lims_p = ()
    
        if pol.lower() == 'spec':
            imvec = img.specvec
            unit = r'$\alpha$'
            factor = 1
            cbar_lims_p = [-5, 5]
            cfun_p = 'jet'
        elif pol.lower() == 'm':
            imvec = img.mvec
            unit = r'$\|\breve{m}|$'
            factor = 1
            cbar_lims_p = [0, 1]
            cfun_p = 'cool'
        elif pol.lower() == 'p':
            imvec = img.mvec * img.ivec
            unit = r'$\|P|$'
            cfun_p = 'afmhot'
        elif pol.lower() == 'chi' or pol.lower() == 'evpa':
            imvec = img.chivec / ehc.DEGREE
            unit = r'$\chi (^\circ)$'
            factor = 1
            cbar_lims_p = [0, 180]
            cfun_p = 'hsv'
        elif pol.lower() == 'e':
            imvec = img.evec
            unit = r'$E$-mode'
            cfun_p = 'Spectral'
        elif pol.lower() == 'b':
            imvec = img.bvec
            unit = r'$B$-mode'
            cfun_p = 'Spectral'
        else:
            pol = pol.upper()
            if pol == 'V':
                cfun_p = 'bwr'
            else:
                cfun_p = 'afmhot'
            try:
                imvec = np.array(img._imdict[pol]).reshape(-1) / (10.**power)
            except KeyError:
                try:
                    if img.polrep == 'stokes':
                        im2 = img.switch_polrep('circ')
                    elif img.polrep == 'circ':
                        im2 = img.switch_polrep('stokes')
                    imvec = np.array(im2._imdict[pol]).reshape(-1) / (10.**power)
                except KeyError:
                    raise Exception("Cannot make pol %s image in display()!" % pol)
    
            unit = fluxunit
            if areaunit != '':
                unit += ' / ' + areaunit
    
        if np.any(np.imag(imvec)):
            print('casting complex image to abs value')
            imvec = np.real(imvec)
    
        imvec = imvec * factor
        imarr = imvec.reshape(img.ydim, img.xdim)
    
        if scale == 'log':
            if (imarr < 0.0).any():
                print('clipping values less than 0 in display')
                imarr[imarr < 0.0] = 0.0
            if log_offset:
                imarr = np.log10(imarr + log_offset / dynamic_range)
            else:
                imarr = np.log10(imarr + np.max(imarr) / dynamic_range)
            unit = r'$\log_{10}$(' + unit + ')'
    
        if scale == 'gamma':
            if (imarr < 0.0).any():
                print('clipping values less than 0 in display')
                imarr[imarr < 0.0] = 0.0
            imarr = (imarr + np.max(imarr) / dynamic_range)**(gamma)
            unit = '(' + unit + ')^' + str(gamma)
    
        if not cbar_lims and cbar_lims_p:
            cbar_lims = cbar_lims_p
    
        if cbar_lims:
            cbar_lims[0] = cbar_lims[0] / (10.**power)
            cbar_lims[1] = cbar_lims[1] / (10.**power)
            imarr[imarr > cbar_lims[1]] = cbar_lims[1]
            imarr[imarr < cbar_lims[0]] = cbar_lims[0]
    
        if has_title:
            plt.title("%s %.2f GHz %s" % (img.source, img.rf / 1e9, pol), fontsize=16)
    
        if not cfun:
            cfun = cfun_p
    
        if cbar_lims:
            im = plt.imshow(imarr, alpha=alpha, cmap=plt.get_cmap(cfun), interpolation=interp,
                            vmin=cbar_lims[0], vmax=cbar_lims[1])
        else:
            im = plt.imshow(imarr, alpha=alpha, cmap=plt.get_cmap(cfun), interpolation=interp)
    
        if not(beamparams is None or beamparams is False):
            beamparams = [beamparams[0], beamparams[1], beamparams[2],
                          -.35 * img.fovx(), -.35 * img.fovy()]
            beamimage = img.copy()
            beamimage.imvec *= 0
            beamimage = beamimage.add_gauss(1, beamparams)
            halflevel = 0.5 * np.max(beamimage.imvec)
            beamimarr = (beamimage.imvec).reshape(beamimage.ydim, beamimage.xdim)
            plt.contour(beamimarr, levels=[halflevel], colors=beamcolor, linewidths=beam_lw)
    
        if has_cbar:
            if only_cbar:
                im.set_visible(False)
            cb = plt.colorbar(im, fraction=0.046, pad=0.04, orientation=cbar_orientation)
            cb.set_label(unit, fontsize=float(cbar_fontsize))
    
            if cbar_fontsize != 12:
                cb.set_label(unit, fontsize=float(cbar_fontsize) / 1.5)
            cb.ax.tick_params(labelsize=cbar_fontsize)
    
            if cbar_lims:
                plt.clim(cbar_lims[0], cbar_lims[1])
            if scinot:
                cb.formatter.set_powerlimits((0, 0))
                cb.update_ticks()
    
    else:  # plot polarization with ticks!
    
        im_stokes = img.switch_polrep(polrep_out='stokes')
        imvec = np.array(im_stokes.imvec).reshape(-1) / (10**power)
        qvec = np.array(im_stokes.qvec).reshape(-1) / (10**power)
        uvec = np.array(im_stokes.uvec).reshape(-1) / (10**power)
        vvec = np.array(im_stokes.vvec).reshape(-1) / (10**power)
    
        if len(imvec) == 0:
            imvec = np.zeros(im_stokes.ydim * im_stokes.xdim)
        if len(qvec) == 0:
            qvec = np.zeros(im_stokes.ydim * im_stokes.xdim)
        if len(uvec) == 0:
            uvec = np.zeros(im_stokes.ydim * im_stokes.xdim)
        if len(vvec) == 0:
            vvec = np.zeros(im_stokes.ydim * im_stokes.xdim)
    
        imvec *= factor
        qvec *= factor
        uvec *= factor
        vvec *= factor
    
        imarr = (imvec).reshape(im_stokes.ydim, im_stokes.xdim)
        qarr = (qvec).reshape(im_stokes.ydim, im_stokes.xdim)
        uarr = (uvec).reshape(im_stokes.ydim, im_stokes.xdim)
        varr = (vvec).reshape(im_stokes.ydim, im_stokes.xdim)
    
        unit = fluxunit
        if areaunit != '':
            unit = fluxunit + ' / ' + areaunit
    
        # only the  stokes I image gets transformed! TODO
        imarr2 = imarr.copy()
        if scale == 'log':
            if (imarr2 < 0.0).any():
                print('clipping values less than 0 in display')
                imarr2[imarr2 < 0.0] = 0.0
            imarr2 = np.log10(imarr2 + np.max(imarr2) / dynamic_range)
            unit = r'$\log_{10}$(' + unit + ')'
    
        if scale == 'gamma':
            if (imarr2 < 0.0).any():
                print('clipping values less than 0 in display')
                imarr2[imarr2 < 0.0] = 0.0
            imarr2 = (imarr2 + np.max(imarr2) / dynamic_range)**(gamma)
            unit = '(' + unit + ')^gamma'
    
        if cbar_lims:
            cbar_lims[0] = cbar_lims[0] / (10.**power)
            cbar_lims[1] = cbar_lims[1] / (10.**power)
            imarr2[imarr2 > cbar_lims[1]] = cbar_lims[1]
            imarr2[imarr2 < cbar_lims[0]] = cbar_lims[0]
    
        # polarization ticks
        m = (np.abs(qvec + 1j * uvec) / imvec).reshape(img.ydim, img.xdim)
    
        thin = img.xdim // nvec
        maska = (imvec).reshape(img.ydim, img.xdim) > pcut * np.max(imvec)
        maskb = (np.abs(qvec + 1j * uvec) / imvec).reshape(img.ydim, img.xdim) > mcut
        mask = maska * maskb
        mask2 = mask[::thin, ::thin]
        x = (np.array([[i for i in range(img.xdim)]
                       for j in range(img.ydim)])[::thin, ::thin])
        x = x[mask2]
        y = (np.array([[j for i in range(img.xdim)]
                       for j in range(img.ydim)])[::thin, ::thin])
        y = y[mask2]
        a = (-np.sin(np.angle(qvec + 1j * uvec) /
                     2).reshape(img.ydim, img.xdim)[::thin, ::thin])
        a = a[mask2]
        b = (
            np.cos(
                np.angle(
                    qvec +
                    1j *
                    uvec) /
                2).reshape(
                img.ydim,
                img.xdim)[
                ::thin,
                ::thin])
        b = b[mask2]
    
        m = (np.abs(qvec + 1j * uvec) / imvec).reshape(img.ydim, img.xdim)
        p = (np.abs(qvec + 1j * uvec)).reshape(img.ydim, img.xdim)
        m[np.logical_not(mask)] = 0
        p[np.logical_not(mask)] = 0
        qarr[np.logical_not(mask)] = 0
        uarr[np.logical_not(mask)] = 0
    
        voi = (vvec / imvec).reshape(img.ydim, img.xdim)
        voi[np.logical_not(mask)] = 0
    
        # Little pol plots
        if plot_stokes:
    
            maxval = 1.1 * np.max((np.max(np.abs(uarr)),
                                   np.max(np.abs(qarr)), np.max(np.abs(varr))))
    
            # P Plot
            ax = plt.subplot2grid((2, 5), (0, 0))
            im = plt.imshow(p, cmap=plt.get_cmap('bwr'), interpolation=interp,
                            vmin=-maxval, vmax=maxval)
            plt.contour(imarr, colors='k', linewidths=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title:
                plt.title('P')
            if has_cbar:
                cbaxes = plt.gcf().add_axes([0.1, 0.2, 0.01, 0.6])
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04, cax=cbaxes,
                                    label=unit, orientation='vertical')
                cbar.ax.tick_params(labelsize=cbar_fontsize)
                cbaxes.yaxis.set_ticks_position('left')
                cbaxes.yaxis.set_label_position('left')
                if cbar_lims:
                    plt.clim(-maxval, maxval)
    
            # V Plot
            ax = plt.subplot2grid((2, 5), (0, 1))
            plt.imshow(varr, cmap=plt.get_cmap('bwr'), interpolation=interp,
                       vmin=-maxval, vmax=maxval)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title:
                plt.title('V')
    
            # Q Plot
            ax = plt.subplot2grid((2, 5), (1, 0))
            plt.imshow(qarr, cmap=plt.get_cmap('bwr'), interpolation=interp,
                       vmin=-maxval, vmax=maxval)
            plt.contour(imarr, colors='k', linewidths=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title:
                plt.title('Q')
    
            # U Plot
            ax = plt.subplot2grid((2, 5), (1, 1))
            plt.imshow(uarr, cmap=plt.get_cmap('bwr'), interpolation=interp,
                       vmin=-maxval, vmax=maxval)
            plt.contour(imarr, colors='k', linewidths=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title:
                plt.title('U')
    
            # V/I plot
            ax = plt.subplot2grid((2, 5), (0, 2))
            im = plt.imshow(voi, cmap=plt.get_cmap('seismic'), interpolation=interp,
                            vmin=-1, vmax=1)
            if has_title:
                plt.title('V/I')
            plt.contour(imarr, colors='k', linewidths=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_cbar:
                cbaxes = plt.gcf().add_axes([0.125, 0.1, 0.425, 0.01])
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04, cax=cbaxes,
                                    label='|m|', orientation='horizontal')
                cbar.ax.tick_params(labelsize=cbar_fontsize)
                cbaxes.yaxis.set_ticks_position('right')
                cbaxes.yaxis.set_label_position('right')
    
                if cbar_lims:
                    plt.clim(-1, 1)
    
            # m plot
            ax = plt.subplot2grid((2, 5), (1, 2))
            plt.imshow(m, cmap=plt.get_cmap('seismic'), interpolation=interp, vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title:
                plt.title('m')
            plt.contour(imarr, colors='k', linewidths=.25)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.01 * img.xdim, units='x', pivot='mid', color='k', angles='uv',
                       scale=1.0 / thin)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.005 * img.xdim, units='x', pivot='mid', color='w', angles='uv',
                       scale=1.1 / thin)
    
            # Big Stokes I plot --axis
            ax = plt.subplot2grid((2, 5), (0, 3), rowspan=2, colspan=2)
        else:
            ax = plt.gca()
    
        if not cfun:
            cfun = 'afmhot'
    
        # Big Stokes I plot
        if cbar_lims:
            im = plt.imshow(imarr2, cmap=plt.get_cmap(cfun), interpolation=interp,
                            vmin=cbar_lims[0], vmax=cbar_lims[1])
        else:
            im = plt.imshow(imarr2, cmap=plt.get_cmap(cfun), interpolation=interp)
    
        if vec_cfun is None:
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.01 * img.xdim, units='x', pivot='mid', color='k', angles='uv',
                       scale=1.0 / thin)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.005 * img.xdim, units='x', pivot='mid', color='w', angles='uv',
                       scale=1.1 / thin)
        else:
            mthin = (
                np.abs(
                    qvec +
                    1j *
                    uvec) /
                imvec).reshape(
                img.ydim,
                img.xdim)[
                ::thin,
                ::thin]
            mthin = mthin[mask2]
            #plt.quiver(x, y, a, b,
            #           headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
            #           width=.01 * img.xdim, units='x', pivot='mid', color='w', angles='uv',
            #           scale=1.0 / thin)
            plt.quiver(x, y, a, b, mthin,
                       norm=mpl.colors.Normalize(vmin=vec_cfun_min, vmax=vec_cfun_max), cmap=vec_cfun,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.007 * img.xdim, units='x', pivot='mid', angles='uv',
                       scale=1.1 / thin)
    
        if not(beamparams is None or beamparams is False):
            beamparams = [beamparams[0], beamparams[1], beamparams[2],
                          -.35 * img.fovx(), -.35 * img.fovy()]
            beamimage = img.copy()
            beamimage.imvec *= 0
            beamimage = beamimage.add_gauss(1, beamparams)
            halflevel = 0.5 * np.max(beamimage.imvec)
            beamimarr = (beamimage.imvec).reshape(beamimage.ydim, beamimage.xdim)
            plt.contour(beamimarr, levels=[halflevel], colors=beamcolor, linewidths=beam_lw)
    
        if has_cbar:
    
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04,
                                label=unit, orientation=cbar_orientation)
            cbar.ax.tick_params(labelsize=cbar_fontsize)
            if cbar_lims:
                plt.clim(cbar_lims[0], cbar_lims[1])
        if has_title:
            plt.title("%s %.1f GHz : m=%.1f%% , v=%.1f%%" % (img.source, img.rf / 1e9,
                                                             img.lin_polfrac() * 100,
                                                             img.circ_polfrac() * 100),
                      fontsize=12)
        f.subplots_adjust(hspace=.1, wspace=0.3)
    
    # Label the plot
    ax = plt.gca()
    if label_type == 'ticks':
        xticks = obsh.ticks(img.xdim, img.psize / ehc.RADPERAS / 1e-6)
        yticks = obsh.ticks(img.ydim, img.psize / ehc.RADPERAS / 1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        plt.xlabel(r'Relative RA ($\mu$as)')
        plt.ylabel(r'Relative Dec ($\mu$as)')
    
    elif label_type == 'scale':
        plt.axis('off')
        fov_uas = img.xdim * img.psize / ehc.RADPERUAS  # get the fov in uas
        roughfactor = 1. / 3.  # make the bar about 1/3 the fov
        fov_scale = int(math.ceil(fov_uas * roughfactor / 10.0)) * 10
        start = img.xdim * roughfactor / 3.0  # select the start location
        end = start + fov_scale / fov_uas * img.xdim  # determine the end location
        plt.plot([start, end], [img.ydim - start - 5, img.ydim - start - 5],
                 color="white", lw=scale_lw)  # plot a line
        plt.text(x=(start + end) / 2.0, y=img.ydim - start + img.ydim / 30,
                 s=str(fov_scale) + r" $\mu$as", color="white",
                 ha="center", va="center", fontsize=scale_fontsize)
        ax = plt.gca()
        if axis is None:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    
    elif label_type == 'none':
        plt.axis('off')
        ax = plt.gca()
        if axis is None:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    
    # Show or save to file
    if axis is not None:
        return axis
    if show:
        plt.show(block=False)
    
    if export_pdf != "":
        f.savefig(export_pdf, bbox_inches='tight', pad_inches=pdf_pad_inches, dpi=dpi)
    
    return f
