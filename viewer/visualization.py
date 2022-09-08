import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from viewer.segment_data import separate_data_chs
from ssqueezepy import ssq_cwt

def plot_random_samples(dataset,n_samples=5,fs=2,n_chs=3,start=None,end=None,type='signal',colormap='Viridis',
                        seed=50):
    if n_samples > len(dataset):
        raise ValueError(f'Number of sample exceeds dataset size. Please enter a number less than or equal {len(dataset)}.')
    np.random.seed(seed)
    if start is None and end is None:
        samples_index = np.random.randint(0,dataset.shape[0], size=n_samples)
    elif start is not None and end is None:
        samples_index = np.random.randint(start,dataset.shape[0], size=n_samples)
    elif start is None and end is not None:
        samples_index = np.random.randint(0,end, size=n_samples)
    else:
        samples_index = np.random.randint(start,end, size=n_samples)

    for index in samples_index:
        if type == 'signal':
            plot_signal(dataset[index],n_chs=n_chs,fs=fs,colormap=colormap)
        elif type == 'spectrogram':
            plot_spectrogram(dataset[index],n_chs=n_chs,fs=fs,colormap=colormap)
        elif type == 'sscwt':
            plot_sscwt_cwt(dataset[index],n_chs=n_chs,fs=fs,colormap=colormap,type=type)
        elif type == 'cwt':
            plot_sscwt_cwt(dataset[index],n_chs=n_chs,fs=fs,colormap=colormap,type=type)
        elif type == 'sscwt/cwt':
            plot_sscwt_cwt(dataset[index],n_chs=n_chs,fs=fs,colormap=colormap,type=type)



def plot_signal(record,n_chs):
    chs = separate_data_chs(record,n_chs)
    
    fig = make_subplots(rows=n_chs, cols=1)

    for i,ch in enumerate(chs):
        fig.append_trace(go.Scatter(
        x=np.arange(0,ch.shape[0]),
        y=ch.flatten(),
        mode='lines',
        name = f'ch{(i+1)}'
        ), row=(i+1), col=1)


    fig.update_xaxes(title_text="Samples")
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(title_text="Signal Channels",margin=dict(r=20, t=40, b=60, l=20),height=700)
    fig.show()

def plot_signals(dataset,start,end,indices=None,n_chs=3):
    # plot choosen signals in subplots
    if indices:
        for i in indices:
            plot_signal(dataset[i],n_chs)
    else:
        for i in range(start,(end+1)):
            plot_signal(dataset[i],n_chs)


def plot_spectrogram(record,n_chs,fs,colormap='Viridis'):
    record_chs = separate_data_chs(record,n_chs)
    fig = make_subplots(rows=n_chs, cols=1)
    for i,ch in enumerate(record_chs): 
        f, t, Sxx = signal.spectrogram(ch.flatten(), fs=fs)
        fig.append_trace(go.Heatmap(
            x=t,
            y = f,
            z=np.array(Sxx),
            colorscale=colormap,

        ),row=(i+1),col=1)
        fig.update_xaxes(title_text="Time segments")
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(title='Spectrogram of signal',margin=dict(r=20, t=40, b=60, l=20),height=700)
        fig.show()


def plot_sscwt_cwt(signal,n_chs=3,fs=2,colormap = 'Viridis',type='sscwt/cwt'):
    record_chs = separate_data_chs(signal,n_chs)
    
    for i,ch in enumerate(record_chs):
        Tx, Wx, *_ = ssq_cwt(ch.reshape(ch.shape[0],))

        if type =='sscwt':
            fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['2D representation of SSCWT', '3D representation of SSCWT'],
                            specs=[[{"type": "heatmap"},{"type": "surface"}]],
                            )
            fig.add_trace(go.Heatmap(
            z=np.abs(Tx),
            texttemplate= "%{z}",
            type = 'heatmap',
            colorscale='Viridis',
            ),row=1,col=1)
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)

            fig.add_trace(go.Surface(z=np.abs(Tx),contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True)),row=1,col=2)
            fig.update_xaxes(title_text="Samples", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_layout(title_text=f"SSCWT Representation of ch{i}")

            fig.show()
        elif type=='cwt':
            fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['2D representation of CWT', '3D representation of CWT'],
                            specs=[[{"type": "heatmap"},{"type": "surface"}]],
                            )
            fig.add_trace(go.Heatmap(
            z=np.abs(Wx),
            texttemplate= "%{z}",
            type = 'heatmap',
            colorscale='Viridis',
            ),row=1,col=1)
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)

            fig.add_trace(go.Surface(z=np.abs(Wx),contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True)),row=1,col=2)
            fig.update_xaxes(title_text="Samples", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_layout(title_text=f"CWT Representation of ch{i}")

            fig.show()

        elif type=='sscwt/cwt':
            fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['2D representation of CWT', '3D representation of CWT'],
                            specs=[[{"type": "heatmap"},{"type": "surface"}]],
                            )
            fig.add_trace(go.Heatmap(
            z=np.abs(Wx),
            texttemplate= "%{z}",
            type = 'heatmap',
            colorscale='Viridis',
            ),row=1,col=1)
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)

            fig.add_trace(go.Surface(z=np.abs(Wx),contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True)),row=1,col=2)
            fig.update_xaxes(title_text="Samples", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_layout(title_text=f"CWT Representation of ch{i}")

            fig.show()

            fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['2D representation of SSCWT', '3D representation of SSCWT'],
                            specs=[[{"type": "heatmap"},{"type": "surface"}]],
                            )
            fig.add_trace(go.Heatmap(
            z=np.abs(Tx),
            texttemplate= "%{z}",
            type = 'heatmap',
            colorscale='Viridis',
            ),row=1,col=1)
            fig.update_xaxes(title_text="Samples", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)

            fig.add_trace(go.Surface(z=np.abs(Tx),contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True)),row=1,col=2)
            fig.update_xaxes(title_text="Samples", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_layout(title_text=f"SSCWT Representation of ch{i}")

            fig.show()




def correlation_map(features_df,colormap='Viridis'):
    corr_df = features_df.corr()
    figure = go.Figure(go.Heatmap(
        x=corr_df.columns,
        y = corr_df.index,
        z=np.array(corr_df),
        texttemplate= "%{z}",
        type = 'heatmap',
        colorscale=colormap,

    ))
    figure.update_layout(title='Correlation Map of Dataset Features')
    figure.show()
    


def plot_mag_response(w,h,fs):
    mag = 20*np.log10(abs(h))
    freq = w*fs/(2*np.pi)
    figure = go.Figure(go.Scatter(x=freq, y=mag,
                mode='lines',
                name='lines'))
    figure.update_layout(title='Magnitude Response',
                xaxis_title='Frequency [Hz]',
                yaxis_title='Magnitude [dB]')
    figure.show()



def plot_phase_response(w,h,fs):
    phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))*(180/np.pi)
    freq = w*fs/(2*np.pi)
    figure = go.Figure(go.Scatter(x=freq, y=phase,
                mode='lines',
                name='lines'))
    figure.update_layout(title='Phase Response',
                xaxis_title='Frequency [Hz]',
                yaxis_title='Phase [degree]')
    figure.show()


def plot_impulse_response(b,a):
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
    response = signal.lfilter(b, a, impulse)

    figure = go.Figure(go.Scatter(x=x, y=response,mode='lines+markers',
                    name='lines+markers'))
    figure.update_layout(title='Impulse Response',
                xaxis_title='Samples',
                yaxis_title='Amplitude',shapes=[dict({
                    'type': 'line',
                    'x0': min(x),
                    'y0': 0,
                    'x1': max(x),
                    'y1': 0,
                    'line': {
                            
                            'width': 2
                        }})])
    figure.show()


def plot_step_response(b,a):
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)
    response = signal.lfilter(b, a, impulse)
    step = np.cumsum(response)
   
    figure = go.Figure(go.Scatter(x=x, y=step,mode='lines+markers',
                    name='lines+markers'))
    figure.update_layout(title='Step Response',
                xaxis_title='Samples',
                yaxis_title='Amplitude',
                shapes=[dict({
                    'type': 'line',
                    'x0': min(x),
                    'y0': 0,
                    'x1': max(x),
                    'y1': 0,
                    'line': {
                            
                            'width': 2
                        }})])
                
    figure.show()

def plot_filter_prop(b,a,w,h,fs):
    plot_mag_response(w,h,fs)
    plot_phase_response(w,h,fs)
    plot_impulse_response(b,a)
    plot_step_response(b,a)

    