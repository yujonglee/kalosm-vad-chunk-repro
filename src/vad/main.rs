use futures_util::StreamExt;
use kalosm_sound::*;

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

    let mut stream = audio_source.transcribe(model);

    while let Some(segment) = stream.next().await {
        println!("[{:?}]: {}", segment.sample_range(), segment.text());
    }
}
