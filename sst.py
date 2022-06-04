import onnx
import torch
import onnxruntime
from omegaconf import OmegaConf
import os

language = 'en' # also available 'de', 'es'

# load provided utils
_, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language=language)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

 # see available models
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
models = OmegaConf.load('models.yml')
available_languages = list(models.stt_models.keys())
assert language in available_languages

# load the actual ONNX model
torch.hub.download_url_to_file(models.stt_models.en.latest.onnx, 'model.onnx', progress=True)
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('model.onnx')


def predictAudio(audio_path):
    """
    audio_path: path to audio file
    """
    # download a single file, any format compatible with TorchAudio
    test_files = [audio_path]
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]))

    # actual onnx inference and decoding
    onnx_input = input.detach().cpu().numpy()
    ort_inputs = {"input": onnx_input}
    ort_outs = ort_session.run(None, ort_inputs)
    decoded = decoder(torch.Tensor(ort_outs[0])[0])
    return decoded
