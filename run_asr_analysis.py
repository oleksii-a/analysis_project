import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchaudio.io import StreamReader
from tqdm import tqdm

#
SEGMENT_LEN = 2560
CONTEXT_LEN = 640
SAMPLE_RATE = 16000


class ContextCacher:
    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(
                chunk, (0, self.segment_length - chunk.size(0))
            )
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()

    files = [os.path.join("audio", filename) for filename in os.listdir("audio")]

    #
    bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    feature_extractor = bundle.get_streaming_feature_extractor()
    decoder = bundle.get_decoder().cuda()
    token_processor = bundle.get_token_processor()

    infer_latencies, feat_latencies, mem_latencies = [], [], []
    num_samples = 0
    for file in tqdm(files):

        cacher = ContextCacher(SEGMENT_LEN, CONTEXT_LEN)
        streamer = StreamReader(file)
        streamer.add_basic_audio_stream(
            frames_per_chunk=SEGMENT_LEN,
            sample_rate=SAMPLE_RATE,
        )

        stream_iterator = streamer.stream()
        state, hypothesis = None, None

        with torch.inference_mode():
            for i, (chunk,) in enumerate(stream_iterator, start=0):
                #
                t_start = time.time()
                segment = cacher(chunk[:, 0])
                features, length = feature_extractor(segment)
                t_end = time.time()
                feat_latencies.append(t_end - t_start)

                #
                t_start = time.time()
                features = features.cuda()
                t_end = time.time()
                mem_latencies.append(t_end - t_start)

                #
                t_start = time.time()
                hypos, state = decoder.infer(
                    features,
                    length,
                    args.beam_width,
                    state=state,
                    hypothesis=hypothesis,
                )
                torch.cuda.synchronize()
                t_end = time.time()
                infer_latencies.append(t_end - t_start)

                hypothesis = hypos

                #
                transcript = token_processor(hypos[0][0], lstrip=False)
                print(transcript)

                #
                num_samples += len(segment)

    #
    total_audio_dur = num_samples / SAMPLE_RATE
    rtf = np.sum(infer_latencies + feat_latencies + mem_latencies) / total_audio_dur

    #
    infer_latency_p95 = np.percentile(infer_latencies, 95)
    feat_latency_p95 = np.percentile(feat_latencies, 95)
    mem_latency_p95 = np.percentile(mem_latencies, 95)
    algo_latency = (SEGMENT_LEN + CONTEXT_LEN) / SAMPLE_RATE

    #
    total_latency = (
        algo_latency + infer_latency_p95 + feat_latency_p95 + mem_latency_p95
    )

    #
    print("Algorithmic latency:", algo_latency)
    print("P95 inference latency:", infer_latency_p95)
    print("P95 feature extraction latency:", feat_latency_p95)
    print("P95 memory latency:", mem_latency_p95)
    print(f"Total latency: {total_latency}")
    print(f"Real-time factor: {rtf}")

    #
    plt.figure()
    plt.grid()
    plt.hist(infer_latencies, bins=50)
    plt.title("Inference latency")
    plt.xlabel("Latency, sec")
    plt.ylabel("Count")

    #
    plt.figure()
    plt.grid()
    plt.hist(feat_latencies, bins=50)
    plt.title("Feature extraction latency")
    plt.xlabel("Latency, sec")
    plt.ylabel("Count")

    plt.show()
