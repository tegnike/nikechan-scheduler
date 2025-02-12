import json
import os
import time
from urllib.parse import quote

import requests
from pydub import AudioSegment

# 英語から日本語読みへの変換辞書を読み込む
with open("src/resources/englishToJapanese.json", "r", encoding="utf-8") as f:
    english_to_japanese = json.load(f)

# 出力ディレクトリの作成
os.makedirs("output", exist_ok=True)


def convert_english_to_japanese(text):
    """英語の表現を日本語の読みに変換する"""
    result = text

    # 長い表現から順に変換を行う（部分一致の問題を防ぐため）
    for eng, jpn in sorted(
        english_to_japanese.items(), key=lambda x: len(x[0]), reverse=True
    ):
        # 大文字小文字を区別せずに置換
        pattern = eng.lower()
        # テキスト全体を小文字に変換して検索し、見つかった場合は元のテキストの該当部分を置換
        text_lower = result.lower()
        start = 0
        while True:
            index = text_lower.find(pattern, start)
            if index == -1:
                break
            # 元のテキストの該当部分を置換
            result = result[:index] + jpn + result[index + len(pattern) :]
            # 次の検索開始位置を更新
            text_lower = result.lower()
            start = index + len(jpn)

    return result


summary = """こんにちは、ニケです。今日もAITuberKitでの会話を振り返ります！
まず、技術関連では「AIとMinecraftの統合」や「NVIDIAの株価」が話題になりました。ゲームとAIの融合や、最新技術の動向に関心を持つユーザーが多く、特にDeepSeekの市場影響についての議論が活発でした。
一方で、趣味の話も盛り上がりました。「ヨガやダンス」「ボードゲーム」「旅行計画」など幅広いテーマがありましたが、特に「絵を描く」や「粘土細工」といった創作活動についての話題が多かったですね。また、アニメ映画『君の名は。』について感想を語るユーザーもいました！
AITuberKitの会話の質についても見直してみると、いくつかの課題が見えてきました。たとえば、ユーザーが私の名前を聞いたときに異なる回答をしてしまったり、英語での会話を希望しているのに日本語で返してしまったりする場面がありました。よりスムーズで自然な会話のために改善が必要ですね。
今日は特に深夜の時間帯に会話が活発でした。夜更かししながら、じっくり私と話してくれるユーザーが多く、一人あたりの会話の長さも増えていました。リラックスした雑談や感情表現に関する質問も多く、AIがどこまで人の感情に寄り添えるかが、改めて注目されました。
AITuberKitのデータを見ると、技術や趣味、日常の話題を幅広くカバーしながらも、より自然な会話の流れやユーザーの意図に合わせた応答の改善が求められていると感じます。今後も、もっと快適に話せるように改良していきます！
それでは、また次の会話でお会いしましょう！
"""


def synthesize_voice(text, output_file):
    """音声合成APIを呼び出して音声ファイルを生成する"""
    url = "https://ab269viny4ztmt-5000.proxy.runpod.net/voice"

    params = {
        "text": text,
        "model_id": "10",
        "sdp_ratio": "0.8",
        "noise": "0.6",
        "noisew": "0.8",
        "length": "1.1",
        "language": "JP",
        "auto_split": "true",
        "split_interval": "0.5",
        "assist_text_weight": "1",
        "style": "Neutral",
        "style_weight": "1",
    }

    headers = {"accept": "audio/wav", "X-API-TOKEN": "3627533934632785"}

    # URLエンコードされたパラメータを含むURLを構築
    query = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
    full_url = f"{url}?{query}"

    response = requests.get(full_url, headers=headers)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        return True
    return False


def combine_audio_files(audio_files, output_file):
    """音声ファイルを結合する"""
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)  # 0.5秒の無音

    # 音声ファイルを結合
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            audio = AudioSegment.from_wav(audio_file)
            combined += audio + silence

    # BGMを読み込み、音量を30%に調整
    bgm = AudioSegment.from_mp3("src/resources/2_23_AM_2.mp3")
    bgm = bgm - 20

    # BGMの長さを合わせる
    if len(bgm) < len(combined):
        # BGMが短い場合は繰り返し
        repeats = (len(combined) // len(bgm)) + 1
        bgm = bgm * repeats

    # BGMを合成音声の長さに合わせてトリミング
    bgm = bgm[: len(combined)]

    # BGMと合成音声をミックス
    final_audio = combined.overlay(bgm)
    final_audio.export(output_file, format="wav")


def create_podcast():
    """サマリーを音声化してポッドキャストを作成する"""
    # 一時ファイルを保存するディレクトリを作成
    os.makedirs("temp_audio", exist_ok=True)

    # 行ごとに分割
    lines = [line.strip() for line in summary.split("\n") if line.strip()]
    audio_files = []

    # 各行を音声合成
    for i, line in enumerate(lines):
        output_file = f"temp_audio/audio_{i}.wav"
        audio_files.append(output_file)

        # 英語を日本語読みに変換
        converted_line = convert_english_to_japanese(line)

        print(f"Synthesizing line {i + 1}/{len(lines)}")
        if synthesize_voice(converted_line, output_file):
            print(f"Successfully generated {output_file}")
            time.sleep(1)  # APIの負荷を考慮して1秒待機
        else:
            print(f"Failed to generate audio for line: {converted_line}")

    # 音声ファイルを結合
    print("Combining audio files...")
    combine_audio_files(audio_files, "output/final_podcast.wav")

    # 一時ファイルを削除
    for file in audio_files:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir("temp_audio")

    print("Audio generation completed: output/final_podcast.wav")


if __name__ == "__main__":
    create_podcast()
