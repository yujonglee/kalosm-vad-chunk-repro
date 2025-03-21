use futures_util::StreamExt;
use kalosm_sound::*;

mod chunker;

#[tokio::main]
async fn main() {
    let cache_dir = dirs::data_dir().unwrap().join("myapp");
    println!("Cache dir: {}", cache_dir.display());

    let audio_source = rodio::Decoder::new_wav(std::io::BufReader::new(
        std::fs::File::open("./data/audio.wav").unwrap(),
    ))
    .unwrap();

    let model = rwhisper::Whisper::builder()
        .with_cache(kalosm_common::Cache::new(cache_dir))
        .with_source(rwhisper::WhisperSource::QuantizedDistilLargeV3)
        .build()
        .await
        .unwrap();

    let chunked =
        crate::chunker::FixedChunkStream::new(audio_source, std::time::Duration::from_secs(7));

    let mut stream =
        rwhisper::TranscribeChunkedAudioStreamExt::transcribe(chunked, model).timestamped();

    while let Some(segment) = stream.next().await {
        println!("[{:?}]: {}", segment.sample_range(), segment.text());
    }
}
