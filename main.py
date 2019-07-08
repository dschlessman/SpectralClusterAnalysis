import lib.SpectralCluster.spectralcluster as sc
import numpy as np

# speech activity detection model trained on AMI training set
SAD_MODEL = ('lib/pyannote-audio/tutorials/models/speech_activity_detection/train/'
             'AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt')

# speaker change detection model trained on AMI training set
SCD_MODEL = ('lib/pyannote-audio/tutorials/models/speaker_change_detection/train/'
             'AMI.SpeakerDiarization.MixHeadset.train/weights/0870.pt')

# speaker embedding model trained on VoxCeleb1
EMB_MODEL = ('lib/pyannote-audio/tutorials/models/speaker_embedding/train/'               
             'VoxCeleb.SpeakerVerification.VoxCeleb1.train/weights/2000.pt')


# one can use their own file like this...
test_file = {'uri': 'filename', 'audio': 'data/130126_009.wav'}

# # ... or use a file provided by a pyannote.database plugin
# # in this example, we are using AMI first test file.s
# from pyannote.database import get_protocol
# from pyannote.database import FileFinder
# preprocessors = {'audio': FileFinder()}
# protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',
#                         preprocessors=preprocessors)
# test_file = next(protocol.test())

#
from pyannote.audio.labeling.extraction import SequenceLabeling
sad = SequenceLabeling(model=SAD_MODEL)
scd = SequenceLabeling(model=SCD_MODEL)

sad_scores = sad(test_file)
#
# # binarize raw SAD scores (as `pyannote.core.Timeline` instance)
# # NOTE: both onset/offset values were tuned on AMI dataset.
# # you might need to use different values for better results.
from pyannote.audio.signal import Binarize
binarize = Binarize(offset=0.94, onset=0.70, log_scale=True)
speech = binarize.apply(sad_scores, dimension=1)
#
# iterate over speech segments (as `pyannote.core.Segment` instances)
for segment in speech:
    print(segment.start, segment.end)

# obtain raw SCD scores (as `pyannote.core.SlidingWindowFeature` instance)
scd_scores = scd(test_file)

# detect peaks and return speaker homogeneous segments
# (as `pyannote.core.Annotation` instance)
# NOTE: both alpha/min_duration values were tuned on AMI dataset.
# you might need to use different values for better results.
from pyannote.audio.signal import Peak
peak = Peak(alpha=0.08, min_duration=0.40, log_scale=True)
partition = peak.apply(scd_scores, dimension=1)
for segment in partition:
    print(segment.start, segment.end)
#
speech_turns = partition.crop(speech)
#
#
# # let's visualize SAD and SCD results using pyannote.core visualization API
# from matplotlib import pyplot as plt
# from pyannote.core import Segment, notebook
#
# # only plot one minute (between t=120s and t=180s)
# notebook.crop = Segment(120, 180)
#
# # helper function to make visualization prettier
# from pyannote.core import SlidingWindowFeature
# plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]), scores.sliding_window)
#
# # create a figure with 6 rows with matplotlib
# nrows = 6
# fig, ax = plt.subplots(nrows=nrows, ncols=1)
# fig.set_figwidth(20)
# fig.set_figheight(nrows * 2)
#
# # 1st row: reference annotation
# # notebook.plot_annotation(None, ax=ax[0])
# ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)
#
# # 2nd row: SAD raw scores
# notebook.plot_feature(plot_ready(sad_scores), ax=ax[1])
# ax[1].text(notebook.crop.start + 0.5, 0.6, 'SAD\nscores', fontsize=14)
# ax[1].set_ylim(-0.1, 1.1)
#
# # 3rd row: SAD result
# notebook.plot_timeline(speech, ax=ax[2])
# ax[2].text(notebook.crop.start + 0.5, 0.1, 'SAD', fontsize=14)
#
# # 4th row: SCD raw scores
# notebook.plot_feature(plot_ready(scd_scores), ax=ax[3])
# ax[3].text(notebook.crop.start + 0.5, 0.3, 'SCD\nscores', fontsize=14)
# ax[3].set_ylim(-0.1, 0.6)
#
# # 5th row: SCD result
# notebook.plot_timeline(partition, ax=ax[4])
# ax[4].text(notebook.crop.start + 0.5, 0.1, 'SCD', fontsize=14)
#
# # 6th row: combination of SAD and SCD
# notebook.plot_timeline(speech_turns, ax=ax[5])
# ax[5].text(notebook.crop.start + 0.5, 0.1, 'speech turns', fontsize=14)


# initialize sequence embedding model
from pyannote.audio.embedding.extraction import SequenceEmbedding
emb = SequenceEmbedding(model=EMB_MODEL, duration=1., step=0.5)

# obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
# embeddings are extracted every 500ms on 1s-long windows
embeddings = emb(test_file)

# for the purpose of this tutorial, we only work of long (> 2s) speech turns
from pyannote.core import Timeline
long_turns = Timeline(segments=[s for s in speech_turns if s.duration > 2.])

def run_speech_pipeline():
    return

def run_spectral_clusterer():
    import numpy as np

    X = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 400 +
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 300 +
        [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]] * 200 +
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 100
    )
    noisy = np.random.rand(1000, 6) * 2 - 1
    X = X + noisy * 0.1

    clusterer =sc.spectral_clusterer.SpectralClusterer(
                p_percentile=0.2,
                gaussian_blur_sigma=0,
                stop_eigenvalue=0.01)

    labels = clusterer.predict(X)

    print(labels)