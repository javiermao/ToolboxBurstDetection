# Copyright (C) 2023
#   Author: Javier M. Antelis <mauricio.antelis@gmail.com>
#   Revision: 1.01
#   Date: 2023/08/14 10:04:11
#   




# -------------------------------
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import copy
from scipy import fftpack

# Import hmmlearn toolbox
from hmmlearn import hmm

# Import Spectral Events toolbox
import spectralevents as se

# Import mne
import mne




# -------------------------------
# Set constanats
dpival = 100




# *****************************************************************************
# *****************************************************************************
# Burst detection methods
# *****************************************************************************
# *****************************************************************************




# -------------------------------
# BurstDetection_THS
def BurstDetection_THS(xraw, fs, FrequencyBand=[15, 29], THRFOM=2, MinSegDur=0, Method='Amplitude', doplot=False):
    """ Detection of bursts segment using the simple threshold (THS) method.
    
    Inputs
    ----------
    xraw: numpy.array of shape (Nsamples,)
        Time-series raw signal
    fs: int
        Sampling frequency. Default is set to 256 Hz
    FrequencyBand: list of length 2
        Low and high frequencies in the band
    THRFOM: int
        Factor-of-the-median threshold to compute the mask
    MinSegDur: int
        Minimum burst/segment duration (i.e., eliminate bursts/segments with duration lower than this)
    Method: str either ´Amplitude´ or ´Power´
        Type of signal used to compute the mask
    doplot: boolean
        Plot the results

    Outputs
    ------
    xmask : numpy.array of shape (Nsamples,)
            Array containing a boolean signal indicating no burst (False) or bursts (True)

    References
    ----------
    [1] 
    
    [2] 
    
    [3] 
    
    """
    
    # -------------------------------
    # Checks
    if MinSegDur<0:
        raise ValueError('PILAS: error in ´BurstDetection_Threshold´, ´MinSegDur´ cannot be negative')
    
    # -------------------------------
    # Create BPF in the specified frequency band
    sos       = signal.butter(5, FrequencyBand, 'bandpass', fs=fs, output='sos')
    
    # -------------------------------
    # Compute filtered signal
    xfil      = signal.sosfiltfilt( sos, xraw )
    
    # -------------------------------
    # Compute signal envelope and amplitude
    xenv      = signal.hilbert( xfil )
    xamp      = np.abs( xenv )
    
    # -------------------------------
    # Use signal amplitude or signal power
    if Method=='Amplitude':
        xamp2     = xamp**1
    elif Method=='Power':
        xamp2     = xamp**2
    
    # -------------------------------
    # Compute threshold
    threshold = THRFOM*np.median( xamp2 )
    
    # -------------------------------
    # Compute mask
    xmask     = CalculateMaskTHS( xamp2, threshold)    
    
    # -------------------------------
    # Eliminate segments smaller than ´MinSegDur´
    if MinSegDur>0:
        xmask      = EliminateShortSegmentsInMask(xmask, MinSegDur, fs, False )    

    # -------------------------------
    # Plot for debugging
    if doplot:
        
        # Compute number of burts from the mask
        SamIni, SamEnd    = CalculateSamplesOfSegmentsInMask(xmask, False )
        Nbursts           = len(SamIni)
        
        # Construct time vector
        t               = np.arange(0,xraw.shape[0],1)/fs
        
        # Plot
        fig, axs = plt.subplots(4, 1, figsize=(8,5), dpi=dpival)

        axs[0].plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_title("There are {} bursts".format(Nbursts), size='large')
        axs[0].set_ylabel('Raw \n signal', size='large')
        axs[0].grid(False)
        # Plot bursts in the raw signal
        for idxini, idxend in zip(SamIni, SamEnd):
            axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)

        axs[1].plot(t, xfil  ,      label='Filtered' , linewidth=2.0, color='r' )
        axs[1].plot(t, xamp  , ':', label='Amplitude', linewidth=1.0, color='g' )
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylabel('Filtered \n signal', size='large')
        axs[1].grid(False)
        
        axs[2].plot(t, xamp2       , label=Method, linewidth=2.0, color='g' )
        axs[2].axhline(y=threshold , label='Threshold', linewidth=1.0, color='k', linestyle='--' )
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylabel(Method+'\n signal', size='large')
        axs[2].grid(False)

        axs[3].plot(t, xmask )
        axs[3].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[3].set_xlim(t[0], t[-1]+1/fs)
        axs[3].set_ylim(-0.02, +1.05)
        axs[3].set_ylabel('Mask'    , size='large')
        axs[3].set_xlabel('Time (s)', size='large')
        axs[3].grid(False)
        
        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return xmask




# -------------------------------
# CalculateMaskThresholdSimple
def CalculateMaskTHS( x, thr):
    
    # -------------------------------
    # Compute mask
    xmask     = x >= thr

    # -------------------------------
    # Return
    return xmask




# -------------------------------
# BurstDetection_THD
def BurstDetection_THD(xraw, fs, FrequencyBand=[15, 29], DualTHRFOM=[ 2, 4 ], MinSegDur=50, Method='Power', doplot=False):

    # -------------------------------
    # Checks
    if MinSegDur<0:
        raise ValueError('PILAS: error in ´BurstDetection_Threshold´, ´MinSegDur´ cannot be negative')   
    
    # -------------------------------
    # Create BPF in the specified frequency band
    sos       = signal.butter(5, FrequencyBand, 'bandpass', fs=fs, output='sos')
    
    # -------------------------------
    # Compute filtered signal
    xfil      = signal.sosfiltfilt( sos, xraw )
    
    # -------------------------------
    # Compute signal envelope and amplitude
    xenv      = signal.hilbert( xfil )
    xamp      = np.abs( xenv )

    # -------------------------------
    # Use signal amplitude or signal power
    if Method=='Amplitude':
        xamp2     = xamp**1
    elif Method=='Power':
        xamp2     = xamp**2
    
    # -------------------------------
    # Compute thresholds
    thrL         = DualTHRFOM[0]*np.median( xamp2 )
    thrH         = DualTHRFOM[1]*np.median( xamp2 )
    DualTHRFOM   = [ thrL, thrH]

    # -------------------------------
    # Compute mask
    xmask, xmaskL, xmaskH = CalculateMaskTHD( xamp2, fs, DualTHRFOM, False )
    if False:
        # Construct time vector
        t               = np.arange(0,xraw.shape[0],1)/fs

        # Compute number of burts from the mask
        SamIni, SamEnd  = CalculateSamplesOfSegmentsInMask(xmask, False )
        Nbursts         = len(SamIni)

        # Plot
        fig, axs = plt.subplots(5, 1, figsize=(8,5), dpi=dpival)

        axs[0].plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_title("There are {} bursts".format(Nbursts), size='large')
        axs[0].set_ylabel('Raw \n signal', size='large')
        axs[0].grid(False)
        # Plot bursts in the raw signal
        for idxini, idxend in zip(SamIni, SamEnd):
            axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)

        axs[1].plot(t, xamp2           , label=Method , linewidth=2.0, color='r' )
        axs[1].axhline(y=DualTHRFOM[1] , label='ThrH' , linewidth=1.0, color='g', linestyle=':')
        axs[1].axhline(y=DualTHRFOM[0] , label='ThrL' , linewidth=1.0, color='m', linestyle='--'  )
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylabel(Method + '\n signal', size='large')
        axs[1].grid(False)

        axs[2].plot(t, xmaskL, color='m' )
        axs[2].fill_between(x=t, y1=xmaskL, alpha=0.3, color='m')
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylim(-0.02, +1.05)
        axs[2].set_ylabel('Mask \n low', size='large')
        axs[2].grid(False)

        axs[3].plot(t, xmaskH, color='g' )
        axs[3].fill_between(x=t, y1=xmaskH, alpha=0.3, color='g')
        axs[3].set_xlim(t[0], t[-1]+1/fs)
        axs[3].set_ylim(-0.02, +1.05)
        axs[3].set_ylabel('Mask \n high', size='large')
        axs[3].grid(False)

        axs[4].plot(t, xmask )
        axs[4].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[4].set_xlim(t[0], t[-1]+1/fs)
        axs[4].set_ylim(-0.02, +1.05)
        axs[4].set_ylabel('Mask', size='large')
        axs[4].set_xlabel('Time (s)', size='large')
        axs[4].grid(False)

        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Eliminate segments smaller than ´MinSegDur´
    if MinSegDur>0:
        xmask      = EliminateShortSegmentsInMask(xmask, MinSegDur, fs, False )    
    
    # -------------------------------
    # Plot for debugging
    if doplot:

        # Compute number of burts from the mask
        SamIni, SamEnd    = CalculateSamplesOfSegmentsInMask(xmask, False )
        Nbursts           = len(SamIni)
        
        # Construct time vector
        t               = np.arange(0,xraw.shape[0],1)/fs
        
        # Plot
        fig, axs = plt.subplots(4, 1, figsize=(8,5), dpi=dpival)

        axs[0].plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_title("There are {} bursts".format(Nbursts), size='large')
        axs[0].set_ylabel('Raw \n signal', size='large')
        axs[0].grid(False)
        # Plot bursts in the raw signal
        for idxini, idxend in zip(SamIni, SamEnd):
            axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)

        axs[1].plot(t, xfil ,      label='Filtered' , linewidth=2.0, color='r' )
        axs[1].plot(t, xamp , ':', label='Amplitude', linewidth=1.0, color='g' )
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylabel('Filtered \n signal', size='large')
        axs[1].grid(False)
        
        axs[2].plot(t, xamp2           , label=Method , linewidth=2.0, color='g' )
        axs[2].axhline(y=DualTHRFOM[1] , label='ThrH' , linewidth=1.0, color='k', linestyle=':')
        axs[2].axhline(y=DualTHRFOM[0] , label='ThrL' , linewidth=1.0, color='k', linestyle='--')
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylabel(Method + ' \n signal', size='large')
        axs[2].grid(False)

        axs[3].plot(t, xmask )
        axs[3].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[3].set_xlim(t[0], t[-1]+1/fs)
        axs[3].set_ylim(-0.02, +1.05)
        axs[3].set_ylabel('Mask'    , size='large')
        axs[3].set_xlabel('Time (s)', size='large')
        axs[3].grid(False)

        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return xmask




# -------------------------------
# CalculateMaskTHD
def CalculateMaskTHD( x, fs, DualTHRFOM, doplot=False ):

    # -------------------------------
    # Get threshols high and low
    thrL               = DualTHRFOM[0]
    thrH               = DualTHRFOM[1]
    
    # -------------------------------
    # Set first and last sample in x to zero to avoid potential code-crash
    x[0]               = 0
    x[-1]              = 0
    
    # -------------------------------
    # Calculate "maskL" (events above thrL) and "maskH" (events above thrH)
    maskL              = np.zeros( x.shape[0] )
    idx                = np.where( x >= thrL)[0]
    maskL[idx]         = 1

    maskH              = np.zeros( x.shape[0] )
    idx                = np.where( x >= thrH)[0]
    maskH[idx]         = 1

    # -------------------------------
    # Compute mask: keep each event in maskL with at least one detection in maskH
    mask               = np.zeros( x.shape[0] )
    
    # Sample Ini and Sample End of all bursts in maskL
    SamIni, SamEnd     = CalculateSamplesOfSegmentsInMask( maskL, False )
    
    # For each event in maskL
    for idxini, idxend in zip(SamIni, SamEnd):
        # Get the values of maskHigh in the current event of maskLow
        maskHtmp = maskH[idxini:idxend]
        # If at least one value in "maskHtmp" is above thrH, then, keep the event
        if np.any(maskHtmp==1):
            mask[idxini:idxend] = 1    

    # -------------------------------
    # Convert masks to booelan
    xmaskL    = maskL.astype(bool)
    xmaskH    = maskH.astype(bool)
    xmask     = mask.astype(bool)

    # -------------------------------
    # Plot for debugging
    if doplot:
        # Construct time vector
        t               = np.arange(0,x.shape[0],1)/fs
        
        # Plot
        fig, axs = plt.subplots(4, 1, figsize=(8,5), dpi=dpival)

        axs[0].plot(t, x  , label='Signal', linewidth=2)
        axs[0].axhline(y=thrH , label='ThrH'    , linewidth=1.0, color='g', linestyle=':')
        axs[0].axhline(y=thrL , label='ThrL'    , linewidth=1.0, color='m', linestyle='--')
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_ylabel('Input \n signal', size='large')
        axs[0].grid(False)

        axs[1].plot(t, xmaskL, color='m' )
        axs[1].fill_between(x=t, y1=xmaskL, alpha=0.3, color='m' )
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylim(-0.02, +1.05)
        axs[1].set_ylabel('Mask \n low', size='large')
        axs[1].grid(False)

        axs[2].plot(t, xmaskH, color='g' )
        axs[2].fill_between(x=t, y1=xmaskH, alpha=0.3, color='g' )
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylim(-0.02, +1.05)
        axs[2].set_ylabel('Mask \n high', size='large')
        axs[2].grid(False)

        axs[3].plot(t, xmask )
        axs[3].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[3].set_xlim(t[0], t[-1]+1/fs)
        axs[3].set_ylim(-0.02, +1.05)
        axs[3].set_ylabel('Mask \n final', size='large')
        axs[3].set_xlabel('Time (s)', size='large')
        axs[3].grid(False)

        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return xmask, xmaskL, xmaskH




# -------------------------------
# BurstDetection_HMM
def BurstDetection_HMM(xraw, fs, FrequencyBand=[15, 29], THRProb=0.5, MinSegDur=0, Method='Amplitude', FixStatesOrder=False, doplot=False):

    # -------------------------------
    # Checks
    if MinSegDur<0:
        raise ValueError('PILAS: error in ´BurstDetection_Threshold´, ´MinSegDur´ cannot be negative')   
    
    # -------------------------------
    # Create BPF in the specified frequency band
    sos       = signal.butter(5, FrequencyBand, 'bandpass', fs=fs, output='sos')
    
    # -------------------------------
    # Compute filtered signal
    xfil      = signal.sosfiltfilt( sos, xraw )
    
    # -------------------------------
    # Compute signal envelope and amplitude
    xenv      = signal.hilbert( xfil )
    xamp      = np.abs( xenv )
    
    # -------------------------------
    # Use signal amplitude or signal power
    if Method=='Amplitude':
        xamp2     = xamp**1
    elif Method=='Power':
        xamp2     = xamp**2

    # -------------------------------
    # Compute masks
    xmask, xprob = CalculateMaskHMM(xamp2, fs, THRProb, False )

    # -------------------------------
    # If the mask stars in True, then change the order of probabilities and recompute the mask
    # Only valid when we know in advance that no burst must initiate in the first time sample
    if FixStatesOrder==True:
        #print("CheckProbOrder")
        if xmask[0]==True:
            #print("The mask stars in",xmask[0] )
            #print("... fixing it")
            xprobold = copy.deepcopy(xprob)
            xprob[:,0] = xprobold[:,1]
            xprob[:,1] = xprobold[:,0]
            xmask      = xprob[:, 1] > THRProb

    # -------------------------------
    # Eliminate segments smaller than ´MinSegDur´
    if MinSegDur>0:
        xmask      = EliminateShortSegmentsInMask(xmask, MinSegDur, fs, False )    

    # -------------------------------
    # Plot for debugging
    if doplot:
        # Compute number of burts from the mask
        SamIni, SamEnd    = CalculateSamplesOfSegmentsInMask(xmask, False )
        Nbursts           = len(SamIni)
        
        # Construct time vector
        t               = np.arange(0,xraw.shape[0],1)/fs

        # Plot
        fig, axs = plt.subplots(4, 1, figsize=(8,5), dpi=dpival)

        axs[0].plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_title("There are {} bursts".format(Nbursts), size='large')
        axs[0].set_ylabel('Raw \n signal', size='large')
        axs[0].grid(False)
        # Plot bursts in the raw signal
        for idxini, idxend in zip(SamIni, SamEnd):
            axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)

        axs[1].plot(t, xfil  ,      label='Filtered' , linewidth=2.0, color='r' )
        axs[1].plot(t, xamp  , ':', label='Amplitude', linewidth=1.0, color='g' )
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylabel('Filtered \n signal', size='large')
        axs[1].grid(False)

        axs[2].plot(t, xprob[:, 1], 'r')
        axs[2].fill_between(x=t, y1=xprob[:, 1], alpha=0.3, color='r')
        axs[2].axhline(y=THRProb, label='THRProb', linewidth=1.0, color='k', linestyle='--')
        axs[2].set_xlim(t[0], t[-1])
        axs[2].set_ylim(-0.02, +1.05)
        axs[2].set_ylabel('Probability \n signal', size='large')
        axs[2].grid(False)

        axs[3].plot(t, xmask )
        axs[3].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[3].set_xlim(t[0], t[-1]+1/fs)
        axs[3].set_ylim(-0.02, +1.05)
        axs[3].set_ylabel('Mask'    , size='large')
        axs[3].set_xlabel('Time (s)', size='large')
        axs[3].grid(False)
        
        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return xmask




# -------------------------------
# CalculateMaskHMM
def CalculateMaskHMM(x, fs, THRProb=0.5, doplot=False ):

    # -------------------------------
    # Set first and last sample in x to zero to avoid potential code-crash
    x[ 0]    = 0
    x[-1]    = 0

    # -------------------------------
    # Apply HMM model several times and save the models and their scores
    SCORES   = list()
    MODELS   = list()

    # Perform several different random starting states
    for idx in range(20):
        # Perform HMM
        model  = hmm.GaussianHMM( n_components=2, covariance_type="full", n_iter=100 )
        model.fit( x.reshape(-1, 1) )

        # Save the current model and its score
        MODELS.append( model )
        SCORES.append( model.score(x[:, None]) )
        #print( f'Converged: {model.monitor_.converged}\t\t' f'Score: {SCORES[-1]}' )

    # -------------------------------
    # Get the best model
    model    = MODELS[ np.argmax(SCORES) ]

    # Predict the most likely sequence of states (probabilities)
    xprob    = model.predict_proba( x.reshape(-1, 1) )

    # -------------------------------
    # Compute mask
    xmask    = xprob[:,1] > THRProb

    # -------------------------------
    # Plot for debugging

    # -------------------------------
    # Return
    return xmask, xprob




# -------------------------------
# BurstDetection_TFE
def BurstDetection_TFE(xraw, fs, FrequencyBand=[15, 29], THRFOM=4.0, MinSegDur=0, freqs=np.array(range(1, 50 + 1)), doplot=False ):

    # -------------------------------
    # Compute TFR
    tfroption=2
    if   tfroption==1:
        # -------------------------------
        # Calculate TFR using SE
        freqs         = list( freqs )
        TFR           = se.tfr( xraw, freqs, fs )
        TFR           = TFR.squeeze()
    elif tfroption==2:
        # -------------------------------
        # Calculate TFR using MNE
        n_cycles      = np.array( freqs )/4
        # Construct a MNE data structure
        info          = mne.create_info( ch_names=['signal'], sfreq=256, ch_types=['misc'] )
        data          = mne.EpochsArray( xraw.reshape(1, -1)[np.newaxis, :, :], info, verbose=False )
        # Use MNE to compute TFR
        TFR           = mne.time_frequency.tfr_morlet(data, freqs=freqs, n_cycles=n_cycles, picks='misc', return_itc=False)
        TFR           = TFR._data.squeeze()

    # -------------------------------
    # Construct time vector
    t                 = np.arange(0,xraw.shape[0],1)/fs

    # -------------------------------
    # Compute mask and infoevents    
    xmask, InfoEvents = CalculateMaskTFR(TFR, t, freqs, fs, FrequencyBand, THRFOM )
    

    # -------------------------------
    # Eliminate segments smaller than ´MinSegDur´
    if MinSegDur>0:
        xmask      = EliminateShortSegmentsInMask(xmask, MinSegDur, fs, False )    
    
    # -------------------------------
    # Compute burst characteristics from info events
    #Bursts_Chars, Bursts_Rate = TB.ComputeBurstCharacteristicsFromInfoEvents(InfoEvents, xraw, fs, FrequencyBand, doplot=False, doprint=False )

    # -------------------------------
    # Compute burst characteristics from mask
    #Bursts_Chars, Bursts_Rate = TB.ComputeBurstCharacteristicsFromMask(xmask, xraw, fs, FrequencyBand, doplot=False, doprint=False )
    
    # -------------------------------
    # Plot for debugging
    if doplot:

        # Compute number of burts from InfoEvents
        #SamIni, SamEnd    = CalculateSamplesOfSegmentsInInfoEvents(InfoEvents, fs)
        SamIni, SamEnd    = CalculateSamplesOfSegmentsInMask(xmask, False )
        Nbursts           = len(SamIni)

        # Plot
        fig, axs = plt.subplots(3, 1, figsize=(8,5))

        axs[0].plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_title("There are {} bursts".format(Nbursts), size='large')
        axs[0].set_ylabel('Raw \n signal', size='large')
        axs[0].grid(False)
        # Plot bursts in the raw signal
        for idxini, idxend in zip(SamIni, SamEnd):
            axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)

        axs[1].pcolormesh(t, freqs, TFR, cmap='RdBu_r', shading='nearest')
        axs[1].axhline(y=FrequencyBand[0], c='w', linewidth=2., linestyle=':', alpha=.7)
        axs[1].axhline(y=FrequencyBand[1], c='w', linewidth=2., linestyle=':', alpha=.7)
        #axs[1].set_yticks([freqs[0], FrequencyBand[0], FrequencyBand[1], freqs[-1]])
        axs[1].set_yticks([1, 10,20,30,40,50])
        axs[1].set_xlim(t[0], t[-1]+0/fs)
        axs[1].set_ylim(freqs[0], freqs[-1])
        axs[1].set_ylabel('Frequency \n (Hz)', size='large')
        axs[1].grid(False)
        # Plot events
        event_times = [event['Peak Time'] for event in InfoEvents]
        event_freqs = [event['Peak Frequency'] for event in InfoEvents]
        alpha = lambda x : (0.6 + 0.4 * np.exp(-0.2 * (x - 30))  # noqa
                            / (1 + np.exp(-0.2 * (x - 30))))  # noqa
        axs[1].scatter(event_times, event_freqs, s=20, c='g', marker='*',
                   alpha=alpha(len(InfoEvents)))

        axs[2].plot(t, xmask )
        axs[2].fill_between(x=t, y1=xmask, alpha=0.3)
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylim(-0.02, +1.05)
        axs[2].set_ylabel('Mask'    , size='large')
        axs[2].set_xlabel('Time (s)', size='large')
        axs[2].grid(False)

        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return xmask, InfoEvents




# -------------------------------
# CalculateMaskTFR
def CalculateMaskTFR(TFR, t, freqs, fs, FrequencyBand, THRFOM ):

    # -------------------------------
    # Find spectral events in "TFR" using Spectral Event Analysis functions
    SpectralEvents  = se.find_events(tfr=TFR,
                                    times=t,
                                    freqs=freqs,
                                    event_band=FrequencyBand,
                                    threshold_FOM=THRFOM)

    # Get the list of the events of the unique existing epoch
    InfoEvents      = SpectralEvents[0]
    
    # -------------------------------
    # Compute mask
    xmask           = ConstructMaskFromInfoEvents(InfoEvents, t, fs, False)

    # -------------------------------
    # Plot for debugging

    # -------------------------------
    # Return
    return xmask, InfoEvents








# *****************************************************************************
# *****************************************************************************
# Compute burst characteristics
# *****************************************************************************
# *****************************************************************************




# -------------------------------
# ComputeBurstCharacteristicsFromMask
def ComputeBurstCharacteristicsFromMask(mask, xraw, fs, FrequencyBand, doplot=False, doprint=False ):
    
    # -------------------------------
    # Initial and end sample of each segment in the mask
    SamIni, SamEnd    = CalculateSamplesOfSegmentsInMask(mask, False )
    
    # -------------------------------
    # Compute burst characteristics from samples
    Bursts_Chars, Bursts_Rate, Bursts_CharsTimes, Bursts_InfoEvents = ComputeBurstCharacteristicsFromSamples(SamIni, SamEnd, xraw, fs, FrequencyBand, doplot, doprint )
    
    # -------------------------------
    # Return
    return Bursts_Chars, Bursts_Rate, Bursts_CharsTimes, Bursts_InfoEvents




# -------------------------------
# ComputeBurstCharacteristicsFromInfoEvents
def ComputeBurstCharacteristicsFromInfoEvents(InfoEvents, xraw, fs, FrequencyBand, doplot=False, doprint=False ):
        
    # -------------------------------
    # Initial and end sample of each segment in the mask
    SamIni, SamEnd    = CalculateSamplesOfSegmentsInInfoEvents(InfoEvents, fs)
    
    # -------------------------------
    # Compute burst characteristics from samples
    Bursts_Chars, Bursts_Rate = ComputeBurstCharacteristicsFromSamples(SamIni, SamEnd, xraw, fs, FrequencyBand, doplot, doprint )
    
    # -------------------------------
    # Return
    return Bursts_Chars, Bursts_Rate




# -------------------------------
# ComputeBurstCharacteristicsFromSamples
def ComputeBurstCharacteristicsFromSamples(SamIni, SamEnd, xraw, fs, FrequencyBand, doplot=False, doprint=False ):
    
    # -------------------------------
    # Construct time vector
    t               = np.arange(0,xraw.shape[0],1)/fs

    # -------------------------------
    # Calculate the "Onset Time" and the "Offset Time" of each segment
    Bursts_OnsetTime  = SamIni/fs
    Bursts_OffsetTime = SamEnd/fs

    Bursts_OnsetTime  = list(Bursts_OnsetTime)
    Bursts_OffsetTime = list(Bursts_OffsetTime)
    
    # -------------------------------
    # Calculate the "Burst rate"
    Nbursts           = len(SamIni)
    SigDur            = len(xraw)/fs
    Bursts_Rate       = Nbursts/SigDur
    
    # -------------------------------
    # Calculate the "Duration" of each burst
    Bursts_Duration   = (SamEnd-SamIni)/fs # s
    Bursts_Duration   = list(Bursts_Duration)

    # -------------------------------
    # Calculate the "Peak amplitude" and the "Peak amplitude time" of each burst

    # Initialize list
    Bursts_PeakAmplitude        = list()
    Bursts_PeakAmplitudeTime    = list()
    
    # Compute filtered signal
    sos      = signal.butter(5, FrequencyBand, 'bandpass', fs=fs, output='sos')
    xfil     = signal.sosfiltfilt( sos, xraw )
    
    # Extract each burst and compute its characteristics
    itmp=0
    for idxini, idxend in zip(SamIni, SamEnd):
        # Get current segment
        ib            = xfil[idxini:idxend]
        it            = t[idxini:idxend]

        # Calculate "Peak amplitude" and the "Peak amplitude time"
        idxmax        = np.argmax( np.abs(ib) )
        iburstampli   = np.abs(ib[idxmax])
        ibursttime    = it[idxmax]

        # Save characteristics
        Bursts_PeakAmplitude.append( iburstampli )
        Bursts_PeakAmplitudeTime.append(  ibursttime )

        # Plot for debugging
        if False:
            plt.figure(figsize=(8, 2))
            plt.plot(t, xfil  , label='Filtered', linewidth=1, alpha=0.5)
            plt.plot(it, ib, label='burst', linewidth=1)
            plt.plot(it, np.abs(ib), label='burst', linewidth=1)
            plt.plot(ibursttime, iburstampli, '*', color='m')
            #for tinitpm, tfintmp in zip(Burst_OnsetTime, Burst_OffsetTime):
            #    plt.axvline(x = tinitpm,  color ='k', linestyle='--')
            #    plt.axvline(x = tfintmp,  color ='k', linestyle='-.')
            plt.axvline(x = Bursts_OnsetTime[itmp],   color ='grey', linestyle=':')
            plt.axvline(x = Bursts_OffsetTime[itmp],  color ='grey', linestyle=':')
            plt.ylabel('Amplitude')
            plt.xlabel('Time (s)')
            plt.grid(True)
            itmp= itmp +1
    
    # -------------------------------
    # Calculate the "Peak frequency" and "Peak frequency magnitude" of each burst

    # Initialize lists
    Bursts_PeakFrequency            = list()
    Bursts_PeakFrequencyMagnitude   = list()

    # Extract each burst and compute its peak frequency using FFT
    for idxini, idxend in zip(SamIni, SamEnd):
        # Get current segment
        ib = xfil[idxini:idxend]
        it = t[idxini:idxend]

        # Calculate "Peak frequency" and "Peak frequency magnitude"
        fmax, Hmax = CalculatePeakFrequencyUsingFFT(it, ib, fs, FrequencyBand, False)

        # Save characteristics
        Bursts_PeakFrequency.append( fmax )
        Bursts_PeakFrequencyMagnitude.append( Hmax )
    
    # -------------------------------
    # Print burst characteristics
    if doprint:
        print("There are:                {} bursts ".format( Nbursts ) )
        print("Burst rate:               {} burst/s ".format( Bursts_Rate ) )
        print(" ")
        
        #print("Onset time:               {} s".format(Bursts_OnsetTime) )      
        #print("Offset time:              {} s".format(Bursts_OffsetTime) )
        print("Duration:                 {} s".format(Bursts_Duration) )
        #print(" ")
    
        print("Peak amplitude:           {} u".format(Bursts_PeakAmplitude) )
        #print("Peak amplitude time:      {} s".format(Bursts_PeakAmplitudeTime) )  
        print(" ")
    
        print("Peak frequency:           {} Hz".format(Bursts_PeakFrequency) )
        print("Peak frequency magnitude: {}".format(Bursts_PeakFrequencyMagnitude) )
        print(" ")

    # -------------------------------
    # Save information in a single array
    #tmp = [Bursts_OnsetTime, Bursts_OffsetTime,  Bursts_Duration, Bursts_PeakAmplitude, Bursts_PeakAmplitudeTime, Bursts_PeakFrequency, Bursts_PeakFrequencyMagnitude]
    
    tmp = [Bursts_Duration, Bursts_PeakAmplitude, Bursts_PeakFrequency, Bursts_PeakFrequencyMagnitude]
    Bursts_Chars=np.array(tmp).T
    #print("There are {} bursts and {} characteristics".format(Bursts_Chars.shape[0],Bursts_Chars.shape[1]) )
    
    #tmp = [Bursts_OnsetTime, Bursts_OffsetTime]
    Bursts_CharsTimes=np.array(tmp).T
    
    
    # -------------------------------
    # Calculate InfoEvents
    Bursts_InfoEvents = list()
    for idx in range(Nbursts):
        # Construct dictonary for the current event
        ThisEvent                              = {}
        ThisEvent['Peak Frequency Magnitude']  = Bursts_PeakFrequencyMagnitude[idx]
        ThisEvent['Peak Frequency']            = Bursts_PeakFrequency[idx]
        ThisEvent['Lower Frequency Bound']     = None
        ThisEvent['Upper Frequency Bound']     = None
        ThisEvent['Frequency Span']            = None
        
        ThisEvent['Peak Amplitude']            = Bursts_PeakAmplitude[idx]
        ThisEvent['Peak Time']                 = Bursts_PeakAmplitudeTime[idx]
        
        ThisEvent['Event Onset Time']          = Bursts_OnsetTime[idx]
        ThisEvent['Event Offset Time']         = Bursts_OffsetTime[idx]        
        ThisEvent['Event Duration']            = Bursts_Duration[idx]
        
        ThisEvent['Peak Power']                = None
        ThisEvent['Normalized Peak Power']     = None
        
        # Save "ThisEvent" in "InfoEvents"
        Bursts_InfoEvents.append( ThisEvent )
    
    # -------------------------------
    # Plot detected bursts
    if doplot:
        plt.figure( figsize=(8,2), dpi=dpival )
        ax  = plt.axes()
        # Raw signal
        ax.plot(t, xraw  , label='Signal', linewidth=2, alpha=0.5)
        # Bursts
        for idxini, idxend in zip(SamIni, SamEnd):
            ax.plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)
        # Bursts ini and end
        #for tinitpm, tfintmp in zip(Bursts_OnsetTime, Bursts_OffsetTime):
        #    ax.axvline(x = tinitpm,  color ='grey', linestyle=':')
        #    ax.axvline(x = tfintmp,  color ='grey', linestyle=':')
        ax.set_xlim(t[0],t[-1]+1/fs)
        ax.set_title("There are {} bursts".format(Nbursts))
        ax.set_xlabel('Time (s)' , size='large')
        ax.set_ylabel('Amplitude', size='large')
        ax.grid(True)
        #ax.legend(loc='upper right')
    
    # -------------------------------
    # Return
    return Bursts_Chars, Bursts_Rate, Bursts_CharsTimes, Bursts_InfoEvents




# -------------------------------
# ComputeBurstCharacteristicsFromEpochs
def ComputeBurstCharacteristicsFromEpochs(X, t, fs, FrequencyBand, method, params, doprint=False):

    # -------------------------------
    # Compute number of epochs
    Nepochs = X.shape[0]

    # -------------------------------
    # Initialize RATE and CHARS
    RATE    = np.zeros( (Nepochs,) )
    CHAR    = np.empty( (0,4) )
    EVENTS  = list()

    # -------------------------------
    # For each epoch
    for iepoch in range(Nepochs):
        
        # Get current epoch
        xraw   = X[iepoch,:]

        # Compute burst characteristics
        if   method=='THS':
            mask                       = BurstDetection_THS(xraw, fs, FrequencyBand, params['THRFOM']    , params['MinSegDur'], 'Amplitude', False )
            chars, rate, _, InfoEvents = ComputeBurstCharacteristicsFromMask(mask, xraw, fs, FrequencyBand, False, False )
        elif method=='THD':
            mask                       = BurstDetection_THD(xraw, fs, FrequencyBand, params['DualTHRFOM'], params['MinSegDur'], 'Power'    , False )
            chars, rate, _, InfoEvents = ComputeBurstCharacteristicsFromMask(mask, xraw, fs, FrequencyBand, False, False )
        elif method=='HMM':
            mask                       = BurstDetection_HMM(xraw, fs, FrequencyBand, params['THRProb']   , params['MinSegDur'], 'Amplitude', True, False)
            chars, rate, _, InfoEvents = ComputeBurstCharacteristicsFromMask(mask, xraw, fs, FrequencyBand, False, False )
        elif method=='TFE':
            mask,_                     = BurstDetection_TFE(xraw, fs, FrequencyBand, params['THRFOM']    , params['MinSegDur'], params['freqs'], False )
            chars, rate, _, InfoEvents = ComputeBurstCharacteristicsFromMask(mask, xraw, fs, FrequencyBand, False, False )
        
        # Print for debugging
        if doprint:
            print("Epoch {} of {} - Rate: {}".format(iepoch,Nepochs,rate) )

        # Save characteristics of the current epoch
        CHAR         = np.append(CHAR, chars, axis=0)
        RATE[iepoch] = rate
        EVENTS.append( InfoEvents )
        
    # -------------------------------
    # Print information
    print("Number of epochs in RATES:   ", RATE.shape[0] )
    print("Number of epochs in EVENTS:  ", len(EVENTS)   )
    print()
    print("Number of events in CHAR:    ", CHAR.shape[0] )
    print("Number of events in EVENTS:  ", sum([len(infoevent) for infoevent in EVENTS]) )
    
    # -------------------------------
    # Return
    return CHAR, RATE, EVENTS








# *****************************************************************************
# *****************************************************************************
# Generate signals
# *****************************************************************************
# *****************************************************************************




# -------------------------------
# GenerateSignalWithBurstsV0
def GenerateSignalWithBurstsV0(fs=256, SignalDuration=2, BurstsParams=0, doplot=False):
    
    # -------------------------------
    # If params is not a dyctionary then set burst parameters
    if BurstsParams==0:        
        BurstsParams             = {}
        BurstsParams['fre']      = 20        # Hz
        BurstsParams['amp']      = 0.4       # units
        BurstsParams['pha']      = np.pi/3   # rad
        BurstsParams['dur']      = 0.200     # s

    # -------------------------------
    # Construct time vector
    Nsamples        = int(SignalDuration*fs)
    t               = np.linspace(0, SignalDuration, Nsamples)

    # -------------------------------
    # Create burst signal (with 4 bursts)

    # Burst onset in time and in samples
    Burst_IniTim    = np.array([0.2, 0.4, 0.6, 0.8]) * SignalDuration   # Seconds
    Burst_IniSam    = np.sort( np.round( Burst_IniTim*fs) ).astype(int) # Samples

    # Burst duration in time and in samples
    burst_dursam    =  np.round( BurstsParams['dur'] * fs ).astype(int)

    # Burst end in samples
    Burst_FinSam    = Burst_IniSam + burst_dursam

    # Create burst signal
    xburst          = np.zeros(Nsamples) # Initalize xburst
    alphatukey      = 0.5                # Shape parameter of the Tukey window
    
    for sampleini, samplefin in zip(Burst_IniSam, Burst_FinSam):
        # Create a single burst
        burstsignal = BurstsParams['amp'] * np.sin(2*np.pi*BurstsParams['fre']*t[sampleini:samplefin] + BurstsParams['pha']) 
        burstsignal = burstsignal * signal.tukey(len(burstsignal), alpha=alphatukey)

        # Add the current burst at the specified onset
        xburst[sampleini:samplefin] = xburst[sampleini:samplefin] + burstsignal
    
    # -------------------------------
    # Create noise

    # Design a LPF
    b, a            = signal.butter(1, 0.5)

    # Create a noise realization
    wnoise          = np.random.random( Nsamples ) - 0.5
    xnoise          = signal.filtfilt(b, a, wnoise) # with an approximately 1/f frequency profile?

    # Compute and plot the power spectrum on the noise
    
    # -------------------------------
    # Create Observation (xnoise + xburst)
    x  =  xnoise  +  xburst
        
    # -------------------------------
    # Plot for debugging
    if doplot:
        
        fig, axs = plt.subplots(4, 1, figsize=(8,6), dpi=dpival)

        ymin,ymax = -0.8, 0.8

        axs[0].plot(t, xnoise  , label='Noise', linewidth=1, alpha=1)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_ylim(ymin,ymax)
        axs[0].set_ylabel('Noise \n signal', size='large')
        axs[0].grid(True)

        axs[1].plot(t, xburst  , label='Burts', linewidth=1, alpha=1)
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylim(ymin,ymax)
        axs[1].set_ylabel('Burst \n signal', size='large')
        axs[1].grid(True)

        axs[2].plot(t, x       , label='Observation', linewidth=1, alpha=1)
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylim(ymin,ymax)
        axs[2].set_ylabel('Observed \n signal', size='large')
        ## Plot bursts in the raw signal
        #for idxini, idxend in zip(Burst_IniSam, Burst_IniSam+Burst_DurSam):
        #    axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)
        axs[2].grid(True)

        # Calculate TFR using MNE
        freqs=np.array(range(1, 50 + 1))
        n_cycles      = np.array( freqs )/4
        # Construct a MNE data structure
        info          = mne.create_info( ch_names=['signal'], sfreq=256, ch_types=['misc'] )
        data          = mne.EpochsArray( x.reshape(1, -1)[np.newaxis, :, :], info, verbose=False )
        # Use MNE to compute TFR
        TFR           = mne.time_frequency.tfr_morlet(data, freqs=freqs, n_cycles=n_cycles, picks='misc', return_itc=False)
        TFR           = TFR._data.squeeze()

        axs[3].pcolormesh(t, freqs, TFR, cmap='RdBu_r', shading='nearest')
        FrequencyBand=[15, 30]
        axs[3].axhline(y=FrequencyBand[0], c='w', linewidth=2., linestyle=':', alpha=.7)
        axs[3].axhline(y=FrequencyBand[1], c='w', linewidth=2., linestyle=':', alpha=.7)
        axs[3].set_yticks([freqs[0], FrequencyBand[0], FrequencyBand[1], freqs[-1]])
        #axs[3].set_yticks([1, 10,20,30,40,50])
        axs[3].set_xlim(t[0], t[-1]+0/fs)
        axs[3].set_ylim(freqs[0], freqs[-1])
        axs[3].set_ylabel('Frequency \n (Hz)', size='large')
        axs[3].grid(False)
        axs[3].set_xlabel('Time (s)'          , size='large')
                
        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return x, t, xburst




# -------------------------------
# GenerateSignalWithBurstsV1
def GenerateSignalWithBurstsV1(fs=256, SignalDuration=1, doplot=False, doprint=False ):
    
    # -------------------------------
    # 1. Construct time vector
    Nsamples        = int(SignalDuration*fs)
    t               = np.linspace(0, SignalDuration, Nsamples)

    # -------------------------------
    # 2. Create burst signal (with 4 bursts)

    # Number of burst
    Nburst          = 4  # Chance according to Burst_Amp and Burst_Fre

    # Burst amplitude
    Burst_Amp       = np.array([0.5, 0.5, 0.5, 0.5])+(np.random.random( Nburst )-0.5)/3

    # Burst frequencies
    Burst_Fre       = np.array([18, 21, 24, 27])       # Hz
    
    # Checks
    if len(Burst_Amp)!=len( Burst_Fre ):
        raise ValueError('PILAS: Check dimensions of ´Burst_Amp´ or ´Burst_Fre´')

    # Burst onset in time and in samples
    Burst_IniTim    = (np.array([0.2, 0.4, 0.6, 0.8]) + np.random.random( Nburst )/20) * SignalDuration   # Seconds
    Burst_IniSam    = np.sort( np.round( Burst_IniTim*fs) ).astype(int)   # Samples

    # Burst duration in time and in samples
    Burst_DurTim    = 70/1000 + np.random.random( Nburst )*(SignalDuration/10) # 50ms + something else depending on SignalDuration
    Burst_DurSam    = np.round( Burst_DurTim * fs ).astype(int)
    
    # Burst phase
    Burst_Pha       =  2*(np.random.random( Nburst )-0.5)*np.pi/2 # [-pi , pi]

    # Create burst signal
    xburst          = np.zeros(Nsamples) # Initalize xburst
    alphatukey      = 0.5                # Shape parameter of the Tukey window, representing the faction of the window inside the cosine tapered region. If zero, the Tukey window is equivalent to a rectangular window. If one, the Tukey window is equivalent to a Hann window.
    
    for amp, fre, pha, sampleini, sampledur in zip(Burst_Amp, Burst_Fre, Burst_Pha, Burst_IniSam, Burst_DurSam):
        # Create a single burst
        burstsignal = amp * np.sin(2*np.pi*fre*t[sampleini:sampleini+sampledur] + pha) 
        burstsignal = burstsignal * signal.tukey(len(burstsignal), alpha=alphatukey)

        # Add the current burst at the specified onset
        xburst[sampleini:sampleini+sampledur] = xburst[sampleini:sampleini+sampledur] + burstsignal
    
    # -------------------------------
    # 3. Create noise

    # Design a LPF
    b, a            = signal.butter(1, 0.5)

    # Create a noise realization
    wnoise          = np.random.random( Nsamples ) - 0.5
    xnoise          = signal.filtfilt(b, a, wnoise) # with an approximately 1/f frequency profile?

    # Compute and plot the power spectrum on the noise
    
    # -------------------------------
    # 4. Create signal (noise + burst)
    x  =  xnoise  +  xburst

    # -------------------------------
    # Plot for debugging
    if doplot:
        
        fig, axs = plt.subplots(4, 1, figsize=(8,6), dpi=dpival)

        ymin,ymax = -0.8, 0.8

        axs[0].plot(t, xnoise  , label='Noise', linewidth=1, alpha=1)
        axs[0].set_xlim(t[0], t[-1]+1/fs)
        axs[0].set_ylim(ymin,ymax)
        axs[0].set_ylabel('Noise \n signal', size='large')
        axs[0].grid(True)

        axs[1].plot(t, xburst  , label='Burts', linewidth=1, alpha=1)
        axs[1].set_xlim(t[0], t[-1]+1/fs)
        axs[1].set_ylim(ymin,ymax)
        axs[1].set_ylabel('Burst \n signal', size='large')
        axs[1].grid(True)

        axs[2].plot(t, x       , label='Observation', linewidth=1, alpha=1)
        axs[2].set_xlim(t[0], t[-1]+1/fs)
        axs[2].set_ylim(ymin,ymax)
        axs[2].set_ylabel('Observed \n signal', size='large')
        ## Plot bursts in the raw signal
        #for idxini, idxend in zip(Burst_IniSam, Burst_IniSam+Burst_DurSam):
        #    axs[0].plot(t[idxini:idxend], xraw[idxini:idxend], linewidth=2)
        axs[2].grid(True)

        # Calculate TFR using MNE
        freqs=np.array(range(1, 50 + 1))
        n_cycles      = np.array( freqs )/4
        # Construct a MNE data structure
        info          = mne.create_info( ch_names=['signal'], sfreq=256, ch_types=['misc'] )
        data          = mne.EpochsArray( x.reshape(1, -1)[np.newaxis, :, :], info, verbose=False )
        # Use MNE to compute TFR
        TFR           = mne.time_frequency.tfr_morlet(data, freqs=freqs, n_cycles=n_cycles, picks='misc', return_itc=False)
        TFR           = TFR._data.squeeze()

        axs[3].pcolormesh(t, freqs, TFR, cmap='RdBu_r', shading='nearest')
        FrequencyBand=[15, 30]
        axs[3].axhline(y=FrequencyBand[0], c='w', linewidth=2., linestyle=':', alpha=.7)
        axs[3].axhline(y=FrequencyBand[1], c='w', linewidth=2., linestyle=':', alpha=.7)
        axs[3].set_yticks([freqs[0], FrequencyBand[0], FrequencyBand[1], freqs[-1]])
        #axs[3].set_yticks([1, 10,20,30,40,50])
        axs[3].set_xlim(t[0], t[-1]+0/fs)
        axs[3].set_ylim(freqs[0], freqs[-1])
        axs[3].set_ylabel('Frequency \n (Hz)', size='large')
        axs[3].grid(False)
        axs[3].set_xlabel('Time (s)'          , size='large')
                
        for ax in axs.flat:
            ax.label_outer()

    # -------------------------------
    # Return
    return x, t, xburst, Burst_Amp, Burst_Fre, Burst_Pha, Burst_IniTim, Burst_DurTim




# -------------------------------
# GenerateEpochsWithBurstsV1
def GenerateEpochsWithBurstsV0(fs=256, SignalDuration=2.0, Nepochs=100, BurstsParams=0, doplot=False ):
    
    # Compute number of samples
    Nsamples        = int(SignalDuration*fs)
    
    # Define number of epochs
    if Nepochs<10:
        Nepochs = 10
    
    # If params is not a dyctionary then set burst parameters
    if BurstsParams==0:        
        BurstsParams             = {}
        BurstsParams['fre']      = 20        # Hz
        BurstsParams['amp']      = 0.4       # units
        BurstsParams['pha']      = np.pi/3   # rad
        BurstsParams['dur']      = 0.200     # s

    # Initialize array for the eeg signals (epochs)
    X             = np.zeros( (Nepochs,Nsamples) ) # Nepochs x Nsamples

    # Generate and save each epochs
    for idx in range(Nepochs):
        # Create epoch
        xraw, t, xburst = GenerateSignalWithBurstsV0(fs, SignalDuration, BurstsParams, doplot=False)

        # Save epoch
        X[idx,:]        = xraw
        
        # Save burst characteristics
        
    if doplot:
        # -------------------------------
        # Plot some of the epochs
        plt.figure(figsize=(8, 4))
        plt.plot(t, X[ 0,:]-4 , linewidth=1)
        plt.plot(t, X[ 1,:]-3 , linewidth=1)
        plt.plot(t, X[ 2,:]-2 , linewidth=1)
        plt.plot(t, X[ 3,:]-1 , linewidth=1)
        plt.plot(t, X[ 4,:]-0 , linewidth=1)
        plt.plot(t, X[ 5,:]+1 , linewidth=1)
        plt.plot(t, X[ 6,:]+2 , linewidth=1)
        plt.plot(t, X[ 7,:]+3 , linewidth=1)
        plt.plot(t, X[ 8,:]+4 , linewidth=1)
        plt.plot(t, X[ 9,:]+5 , linewidth=1)
        plt.grid()
        plt.title('Plot of the first ten epochs')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.xlim(t[0],t[-1])
        #plt.tight_layout()
        
        # Compute X average
        Xavg = np.mean(X, axis=0)
        
        # -------------------------------
        # Plot the average across-all-epochs
        plt.figure(figsize=(8, 2))
        plt.plot(t, Xavg, linewidth=1, label='Average' )
        plt.grid()
        plt.title('Average')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.ylim(-1,+1)
        plt.xlim(0-0.1,SignalDuration+0.1)
        #plt.tight_layout()

    # -------------------------------
    # Return
    return X, t








# *****************************************************************************
# *****************************************************************************
# Complementary functions
# *****************************************************************************
# *****************************************************************************




# -------------------------------
# CalculateSamplesOfSegmentsInMask
def CalculateSamplesOfSegmentsInMask(mask, doplot ):
    """
    #mask = np.array([False, False, False, True, True, True, False, False, False, True, True, True, False, False, False ])
    #mask = np.array([True, True, True, False, False, False, True, True, True, False, False, False ])
    #mask = np.array([False, False, False, True, True, True, False, False, False, True, True, True])

    """

    # -------------------------------
    # Plot for debugging    
    if False:        
        nmask          = np.arange(0,len(mask),1)
        plt.figure(figsize=(8, 4))    
        plt.plot(nmask, mask, 'g')
        plt.fill_between(x=nmask, y1=mask, alpha=0.3, color='g')
        plt.ylabel('Mask')
        plt.xlabel('n')
        plt.grid()
    
    # -------------------------------
    # Convert mask to 0´s and 1's
    mask           = 1*mask
    nmask          = np.arange(0,len(mask),1)
    
    # -------------------------------
    # Compute mask diff where +1 indicates change from 0 to 1 (mask initiation) and -1 indicates change from 1 to 0 (mask end)
    maskdiff       = np.diff(mask)
    idxmaskdiff    = np.arange(0,len(maskdiff),1)
    
    # -------------------------------
    # Initialize lists form samples ini and samples end
    SamIni = []
    SamEnd = []
    
    # -------------------------------
    # For each value in maskdiff ...
    for idx, imaskdiff in zip(idxmaskdiff,maskdiff):    
        if   imaskdiff==+1: # identify mask initiation
            SamIni.append(idx+1)
        elif imaskdiff==-1: # identify mask finalization
            SamEnd.append(idx+0)
    
    # -------------------------------
    # Convert lists to arrays
    SamIni = np.array( SamIni ).astype(int)
    SamEnd = np.array( SamEnd ).astype(int)
    
    # -------------------------------
    #print("From CalculateSamplesOfSegmentsInMask L1")
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()
    
    # -------------------------------
    # Fix if the first masked segment starts prior to sample 0
    if len(SamIni)<len(SamEnd) and mask[ 0]==1: SamIni = np.concatenate( (np.array([0]),SamIni) )
    # Fix if the last masked segment finalizes after sample -1
    if len(SamEnd)<len(SamIni) and mask[-1]==1: SamEnd = np.concatenate( (SamEnd,np.array([len(mask)-1])) )
    # Fix if the first masked segment starts prior to sample 0 and  the last masked segment finalizes after sample -1
    if mask[ 0]==1 and mask[-1]==1:
        SamIni = np.concatenate( (np.array([0]),SamIni) )
        SamEnd = np.concatenate( (SamEnd,np.array([len(mask)-1])) )
    
    # -------------------------------
    #print("From CalculateSamplesOfSegmentsInMask L2")
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()
    
    # -------------------------------
    # Check that all is ok
    if len(SamIni)!=len(SamEnd):
        raise ValueError('PILAS: Check dimensions of ´SamIni´ and ´SamEnd´')
    
    # -------------------------------
    # Verificar que la diferencia SamEnd-SamIni sea siempre positiva
    DurInSam = SamEnd-SamIni
    if np.any(DurInSam < 0):
        raise ValueError('PILAS: Check the difference between ´SamIni´ and ´SamEnd´ (no puede haber negativos)')
    
    # -------------------------------
    # Plor for debugging
    if doplot:
        
        print("The total number of segments is {} ".format( len(SamIni) ) )
        
        plt.figure(figsize=(8, 4))
    
        plt.plot(nmask, mask, 'g')
        plt.fill_between(x=nmask, y1=mask, alpha=0.3, color='g')
    
        for idx in SamIni: plt.axvline(x = idx, linestyle='-.')
        for idx in SamEnd: plt.axvline(x = idx, linestyle='--')
    
        plt.ylabel('Mask')
        plt.xlabel('n')
        plt.grid()
    
    # -------------------------------
    # Return
    return SamIni, SamEnd




# -------------------------------
# CalculateSamplesOfSegmentsInInfoEvents
def CalculateSamplesOfSegmentsInInfoEvents(InfoEvents, fs):
    
    # -------------------------------
    # Calculate Nburst
    Nburst            = len(InfoEvents)

    # -------------------------------
    # Initial and end sample of each segment in InfoEvents
    TimIni = np.zeros( (Nburst,) ) 
    TimEnd = np.zeros( (Nburst,) )
    for idx in range(Nburst):
        TimIni[idx]   = InfoEvents[idx]['Event Onset Time']
        TimEnd[idx]   = InfoEvents[idx]['Event Offset Time']
    SamIni = TimIni*fs
    SamEnd = TimEnd*fs
    SamIni = SamIni.astype(int)
    SamEnd = SamEnd.astype(int)
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()    
    
    # -------------------------------
    # Return
    return SamIni, SamEnd




# -------------------------------
# EliminateShortSegmentsInMask
def EliminateShortSegmentsInMask(mask, MinSegDur, fs, doplot ):
    
    # -------------------------------
    # Compute the initial and end sample of each segment in the mask 
    SamIni, SamEnd = CalculateSamplesOfSegmentsInMask(mask, False )
    #print("From EliminateShortSegmentsInMask")
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()
    
    # -------------------------------
    # Eliminate samples of short segments
    SamIniNew, SamEndNew = EliminateSamplesOfShortSegments(SamIni, SamEnd, MinSegDur, fs )
    #print("From EliminateSamplesOfShortSegments")
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()
    
    # -------------------------------
    # Construct mask from samples
    mask_new = ConstructMaskFromSamples(mask, SamIniNew, SamEndNew)
    
    # -------------------------------
    # Plot for debugging
    if doplot:
        plt.figure(figsize=(8, 6))
        
        plt.subplot(211)
        plt.plot(mask)
        plt.ylabel('Mask')
        
        plt.subplot(212)
        plt.plot(mask_new )
        plt.ylabel('New mask')
        plt.xlabel('Time (s)')

    # -------------------------------
    # Return
    return mask_new




# -------------------------------
# EliminateSamplesOfShortSegments
def EliminateSamplesOfShortSegments(SamIni, SamEnd, MinDur, fs ):

    # -------------------------------
    # Calculate the time duration of each segment 
    BurstDuration = 1000*(SamEnd-SamIni)/fs # ms
    
    # -------------------------------
    # Identify short segments (lower that 100ms)
    shortburst    = BurstDuration<MinDur
    idxshortburst = np.arange(0,len(shortburst),1)
    
    # -------------------------------
    # Initialize lists form new samples ini and samples end
    NewSamIni = []
    NewSamEnd = []
    
    for idx,ishortburst in  zip(idxshortburst,shortburst):
        # Keep non short segments
        if not ishortburst:
            NewSamIni.append( SamIni[idx] )
            NewSamEnd.append( SamEnd[idx] )
    
    # -------------------------------
    # Convert lists to arrays
    NewSamIni = np.array( NewSamIni )
    NewSamEnd = np.array( NewSamEnd )
    
    # -------------------------------
    # Verify that the difference SamEnd-SamIni is always possitive
    DurInSam = NewSamEnd-NewSamIni
    if np.any(DurInSam < 0):
        raise ValueError('PILAS: Check NewSamIni and NewSamEnd (no puede haber negativos)')
        
    # -------------------------------
    # Return
    return NewSamIni, NewSamEnd




# -------------------------------
# ConstructMaskFromSamples
def ConstructMaskFromSamples(mask, SamIni, SamEnd ):
    
    # -------------------------------
    # Construct a new mask
    mask_new = False*mask
    
    #print("*********************************")
    #print("Input to ConstructMaskFromSamples")
    #print("SamIni", SamIni)
    #print("SamEnd", SamEnd)
    #print()
    
    #print("mask_new.shape", mask_new.shape)
    #print("unique(mask_new)", np.unique(mask_new))
    #print("mask_new", mask_new)
    #print()
    
    
    # -------------------------------
    # Poner en True los segments given by [SamIni, SamEnd]
    for idxini,idxfin in  zip(SamIni, SamEnd):
        mask_new[idxini:idxfin+1] = True
    
    # -------------------------------
    # Return
    return mask_new




# -------------------------------
# MergeNeighboringSegments
def MergeNeighboringSegments(mask, t, fs, MinSep, doplot ):
    # -------------------------------
    # Compute the initial and end sample of each segment in the mask
    SamIni, SamEnd = CalculateSamplesOfSegmentsInMask(mask, False )
    
    # -------------------------------
    # Calculate the total number of burst
    Nbursts        = len(SamIni)
    mask2          = copy.copy(mask)
    
    # -------------------------------
    # Calcular la separacion entre segmentos consecutivos y merge if duration<25ms
    for idx in range(Nbursts-1):
    
        # Get info for segment i and i+1
        #dura0      = 1000*(SamEnd[idx  ]-SamIni[idx  ] )/fs # Duration of segment i
        #dura1      = 1000*(SamEnd[idx+1]-SamIni[idx+1] )/fs # Duration of segment i+1
        sepa       = 1000*(SamIni[idx+1]-SamEnd[idx  ] )/fs # Separation between consecutive segments
        #print(idx,dura0,dura1,sepa)
    
        # If separation is lower than 25ms, then merge segments
        if sepa<MinSep:
            #print(mask2[ SamEnd[idx]:SamIni[idx+1]+1 ])
            mask2[ SamEnd[idx]:SamIni[idx+1]+1 ] = True
            #print(mask2[ SamEnd[idx]:SamIni[idx+1]+1 ])
    
    # -------------------------------
    # Print for check
    #SamIni, SamEnd = CalculateSamplesOfSegmentsInMask(mask, False )
    #print(SamIni)
    #print(SamEnd)
    
    #SamIni2, SamEnd2 = CalculateSamplesOfSegmentsInMask(mask2, False )
    #print(SamIni2)
    #print(SamEnd2)
    
    # -------------------------------
    # Plot mask before and after
    if doplot:
        plt.figure(figsize=(8, 5))
    
        plt.subplot(211)
        plt.plot(t, mask )
        plt.fill_between(x=t, y1=mask, alpha=0.3)
        plt.ylabel('Burst \n segments')
    
        plt.subplot(212)
        plt.plot(t, mask2 )
        plt.fill_between(x=t, y1=mask2, alpha=0.3)
        plt.ylabel('Burst \n segments')
        plt.xlabel('Time (s)')
    
    # -------------------------------
    # Return
    return mask2




# -------------------------------
# ConstructMaskFromInfoEvents
def ConstructMaskFromInfoEvents(SpectralEvents, times, fs, doplot=False):
    # Initalize lists    
    SamIni      = list()
    SamEnd      = list()

    # For each event:
    for ievent, event in enumerate(SpectralEvents):
        #print("Event {} of {} ".format(ievent,len(SpectralEvents)) )

        # Get SamIni and SamEnd
        SamIni.append( int(event['Event Onset Time']*fs) )
        SamEnd.append( int(event['Event Offset Time']*fs) )
    
    # Convert lists to arrays
    SamIni      = np.array( SamIni )
    SamEnd      = np.array( SamEnd )
    #print(SamIni)
    #print(SamEnd)

    # Sort in ascending order
    idxsorted   = np.argsort(SamIni)
    SamIni      = SamIni[idxsorted]
    SamEnd      = SamEnd[idxsorted]    
    #print(SamIni)
    #print(SamEnd)

    # -------------------------------
    # Construct mask from samples
    MaskInFalse = np.full( times.shape, False)
    Mask        = ConstructMaskFromSamples(MaskInFalse, SamIni, SamEnd)

    # -------------------------------
    # Verify that the difference SamEnd-SamIni is always possitive
    DurInSam = SamEnd-SamIni
    if np.any(DurInSam < 0):
        raise ValueError('PILAS: Check SamIni and SamEnd (no puede haber negativos)')

    # -------------------------------
    # Plot for debugging
    if doplot:
        plt.figure(figsize=(8, 4))
    
        plt.plot(times, Mask, 'g')
        plt.fill_between(x=times, y1=Mask, alpha=0.3, color='g')    
        plt.ylabel('Mask')
        plt.xlabel('Time (s)')
        plt.grid()
        #plt.tight_layout()

    # -------------------------------
    # Return
    return Mask




# -------------------------------
# Computefft
def Computefft(t, x, fs, doplot=False):

    # -------------------------------
    # Compute delta t
    dt = t[1] - t[0]

    # -------------------------------
    #Set Nfft
    if t.shape[0]<fs:
        Nfft = fs
    else:
        Nfft = t.shape[0]
    
    # -------------------------------
    # Compute FFT
    #H = dt*fftpack.fft(x)
    #f = fftpack.fftfreq(len(t),dt)
    H = dt*fftpack.fft(x,Nfft)
    f = fftpack.fftfreq(Nfft,dt)
    
    # -------------------------------
    # Shift H and f into normal order
    f = fftpack.fftshift(f)
    H = fftpack.fftshift(H)
    
    # Keep only FFT from n/2+1 to n-1 or $f \in (0,\pi=f_s/2]$, that is, double-sided to one-sided
    n = len(f)
    f = f[int(n/2)+1:n-1]
    H = 2.0*H[int(n/2)+1:n-1]

    # -------------------------------
    # Compute magnitude
    Hmag = np.abs(H)

    # -------------------------------
    # Plot for debugging
    if doplot:
        plt.figure(figsize=(8, 4))

        plt.subplot(2,1,1)
        plt.plot(t, x, label='x(t)', linewidth=1)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(f, Hmag  , label='FFT', linewidth=1, alpha=0.99)
        plt.ylabel('|H(f)|')
        plt.xlabel('Frequency (Hz)')
    
    # -------------------------------
    # Return
    return f, H, Hmag # Type: <class 'numpy.ndarray'> ; Shape: (Npoints,)




# -------------------------------
# CalculatePeakFrequencyUsingFFT
def CalculatePeakFrequencyUsingFFT(t, x, fs, FrequencyBand, doplot=False):

    # -------------------------------
    # Compute FFT
    f, _, H   = Computefft(t, x, fs, False)

    # -------------------------------
    # Get indexes and values of the FrequencyBand
    idxl      = np.argwhere(f < FrequencyBand[0])
    idxl      = idxl[-1][0]+1
    #print("Frequency {0:.2f} corresponds to index {1:d}".format(f[idxl], idxl) )

    idxh      = np.argwhere(f > FrequencyBand[1])
    idxh      = idxh[0][0]
    #print("Frequency {0:.2f} corresponds to index {1:d}".format(f[idxh], idxh) )

    fband     = f[idxl:idxh]
    Hband     = H[idxl:idxh]

    # Get PeakFrequency
    idxmax    = np.argmax(Hband)
    fmax      = fband[idxmax]
    Hmax      = Hband[idxmax]
    #print("The maximum of {0:.2f} is in the frequency {1:.2f}".format(Hmax, fmax) )

    # -------------------------------
    # Plot for debugging
    if doplot:
        plt.figure(figsize=(8, 4))

        plt.subplot(2,1,1)
        plt.plot(t, x, label='x(t)', linewidth=1)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(f, H, label='|FFT|', linewidth=1, alpha=0.99)
        plt.axvline(x = fband[0],  color ='k', linestyle=':')
        plt.axvline(x = fband[-1], color ='k', linestyle=':', label='Frequency band')
        plt.plot(fmax, Hmax, '*', color='m', label="{0:.2f}Hz".format(fmax))
        plt.ylabel('|H(f)|')
        plt.xlabel('Frequency (Hz)')
        plt.legend()
        plt.xlim(0,fs/2)

    # -------------------------------
    # Return
    return fmax, Hmax




