from pydub import AudioSegment

class Converter:
    def to_mp3(self, src_path: str, dst_path: str):
        try:
            audio = AudioSegment.from_file(src_path)
            audio.export(dst_path, format="mp3", bitrate="192k")
            print(f"Converted to MP3: {dst_path}")
        except Exception as e:
            raise RuntimeError(f"MP3 Conversion Error: {e}")


if __name__ == "__main__":
    converter = Converter()
    source_file = "files/videoplayback.mp4"
    destination_file = "files/videoplayback2.mp3"
    converter.to_mp3(source_file, destination_file)
        