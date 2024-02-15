import numpy as np
import ehtim as eh

from imagingbase.ehtim_utils import coh_avg_vis

class Calibrator:
    def __init__(self, obs, zero_baseline, zbl, uv_zblcut, reverse_taper_uas, sys_noise, dt=0.0165, **kwargs):  
        self.obs = obs
        
        self.zero_baseline_a1 = zero_baseline[0]
        self.zero_baseline_a2 = zero_baseline[1]
        self.zbl = zbl
        self.uv_zblcut = uv_zblcut
        self.reverse_taper_uas = reverse_taper_uas
        self.sys_noise = sys_noise
        
        self.dt = dt
        
        self.err_type = kwargs.get('err_type', 'predicted')
        self.seed = kwargs.get('seed', 12)
        
    def prepare_data(self):
        # scan-average the data
        # identify the scans (times of continous observation) in the data
        self.obs.add_scans(dt=self.dt)

        # coherently average the scans, which can be averaged due to ad-hoc phasing
        #self.obs = self.obs.avg_coherent(0.,scan_avg=True)
        self.obs = self.avg_coherent(0, scan_avg=True, err_type=self.err_type)

        # Estimate the total flux density from the zero baseline
<<<<<<< Updated upstream
        zbl_tot   = np.median(np.asarray(self.obs.unpack_bl(self.zero_baseline_a1,self.zero_baseline_a2,'amp'), dtype=np.dtype('float,float'))['f1'])#['amp'])
=======
        zbl_tot   = np.median(np.asarray(self.obs.unpack_bl(self.zero_baseline_a1,self.zero_baseline_a2,'amp'),dtype=np.dtype('float,float'))['f1'])#[1]#['amp'])
>>>>>>> Stashed changes
        if self.zbl > zbl_tot:
            print('Warning: Specified total compact flux density ' +
                  'exceeds total flux density')

        # Flag out sites in the obs.tarr table with no measurements
        allsites = set(self.obs.unpack(['t1'])['t1'])|set(self.obs.unpack(['t2'])['t2'])
        self.obs.tarr = self.obs.tarr[[o in allsites for o in self.obs.tarr['site']]]
        self.obs = eh.obsdata.Obsdata(self.obs.ra, self.obs.dec, self.obs.rf, self.obs.bw, self.obs.data, self.obs.tarr,
                                 source=self.obs.source, mjd=self.obs.mjd,
                                 ampcal=self.obs.ampcal, phasecal=self.obs.phasecal)

        # Rescale short baselines to excize contributions from extended flux
        if self.zbl != zbl_tot:
            self.rescale_zerobaseline(self.obs, self.zbl, zbl_tot, self.uv_zblcut)

        # Order the stations by SNR.
        # This will create a minimal set of closure quantities
        # with the highest snr and smallest covariance.
        self.obs.reorder_tarr_snr()
        
        
    def avg_coherent(self, inttime, scan_avg=False, moving=False, err_type='predicted', seed=12):
        """Coherently average data along u,v tracks in chunks of length inttime (sec)

           Args:
                inttime (float): coherent integration time in seconds
                scan_avg (bool): if True, average over scans in self.scans instead of intime
                moving (bool): averaging with moving window (boxcar width in seconds)
           Returns:
                (Obsdata): Obsdata object containing averaged data
        """

        if (scan_avg) and (getattr(self.obs.scans, "shape", None) is None or len(self.obs.scans) == 0):
            print('No scan data, ignoring scan_avg!')
            scan_avg = False

        if inttime <= 0.0 and scan_avg is False:
            print('No averaging done!')
            return self.obs.copy()


        vis_avg = coh_avg_vis(self.obs, dt=inttime, return_type='rec',
                                       err_type=err_type, scan_avg=scan_avg, seed=seed)

        arglist, argdict = self.obs.obsdata_args()
        arglist[4] = vis_avg
        out = eh.obsdata.Obsdata(*arglist, **argdict)

        return out
        
    # Rescale short baselines to excise contributions from extended flux.
    # setting zbl < zbl_tot assumes there is an extended constant flux component of zbl_tot-zbl Jy
    def rescale_zerobaseline(self, obs, totflux, orig_totflux, uv_max):
        multiplier = totflux / orig_totflux
        for j in range(len(obs.data)):
            if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 >= uv_max: continue
            for field in ['vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma']:
                obs.data[field][j] *= multiplier
                
    def precalibrate_data(self):
        self.obs_sc = self.obs.copy() # From here on out, don't change obs. Use obs_sc to track gain changes
        
        # Reverse taper the observation: this enforces a maximum resolution on reconstructed features
        if self.reverse_taper_uas > 0:
            self.obs_sc = self.obs_sc.reverse_taper(self.reverse_taper_uas*eh.RADPERUAS)
        
        # Add non-closing systematic noise to the observation
        self.obs_sc = self.obs_sc.add_fractional_noise(self.sys_noise)
        
        self.obs_sc_init = self.obs_sc.copy()
        
        

