"""
動画から指定秒数のフレームを切り出すモジュール
"""
import cv2
import os
import argparse
from pathlib import Path

def extract_frame_at_time(video_path, time_seconds, output_path=None):
    """
    動画から指定した秒数のフレームを切り出す
    
    Args:
        video_path (str): 入力動画ファイルパス（.mkvファイル）
        time_seconds (float): 切り出したい時間（秒）
        output_path (str, optional): 出力ファイルパス。Noneの場合は自動生成
        
    Returns:
        tuple: (success, frame_array, output_file_path)
               success: 成功したかどうか (bool)
               frame_array: フレーム画像のnumpy配列
               output_file_path: 保存されたファイルのパス
    """
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return False, None, None
    
    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"動画情報: FPS={fps:.2f}, 総フレーム数={total_frames}, 長さ={duration:.2f}秒")
    
    # 指定時間が動画の長さを超えていないかチェック
    if time_seconds > duration:
        print(f"エラー: 指定時間({time_seconds}秒)が動画の長さ({duration:.2f}秒)を超えています")
        cap.release()
        return False, None, None
    
    # 指定時間のフレーム番号を計算
    target_frame = int(time_seconds * fps)
    
    # 指定フレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # フレームを読み込み
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"エラー: フレームの読み込みに失敗しました（{time_seconds}秒, フレーム{target_frame}）")
        return False, None, None
    
    # 出力パスが指定されていない場合は自動生成
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"../data/output/frame_{video_name}_{time_seconds:06.2f}s.jpg"
    
    # 出力ディレクトリを作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # フレームを保存
    success = cv2.imwrite(output_path, frame)
    
    if success:
        print(f"フレーム切り出し成功: {time_seconds}秒 -> {output_path}")
        return True, frame, output_path
    else:
        print(f"エラー: フレームの保存に失敗しました: {output_path}")
        return False, frame, None

def extract_frames_at_intervals(video_path, interval_seconds, output_dir="../data/output/frames"):
    """
    動画から指定間隔でフレームを切り出す
    
    Args:
        video_path (str): 入力動画ファイルパス
        interval_seconds (float): 切り出し間隔（秒）
        output_dir (str): 出力ディレクトリ
        
    Returns:
        list: 切り出されたフレームファイルのパスリスト
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return []
    
    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    cap.release()
    
    # 切り出し時間のリストを生成
    times = []
    current_time = 0
    while current_time < duration:
        times.append(current_time)
        current_time += interval_seconds
    
    # 各時間でフレームを切り出し
    output_files = []
    video_name = Path(video_path).stem
    
    for i, time_sec in enumerate(times):
        output_path = f"{output_dir}/{video_name}_frame_{i:04d}_{time_sec:06.2f}s.jpg"
        success, _, file_path = extract_frame_at_time(video_path, time_sec, output_path)
        if success:
            output_files.append(file_path)
    
    print(f"合計 {len(output_files)} フレームを切り出しました")
    return output_files

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='動画から指定秒数のフレームを切り出す')
    parser.add_argument('video_file', help='動画ファイル名（data/input/内のファイル名）')
    parser.add_argument('time_seconds', type=float, help='切り出したい時間（秒）')
    parser.add_argument('--output', '-o', help='出力ファイルパス（オプション）')
    parser.add_argument('--interval', '-i', type=float, help='間隔モード: 指定秒間隔で複数フレーム切り出し')
    
    args = parser.parse_args()
    
    # 動画ファイルのフルパスを構築
    video_path = f"../data/input/{args.video_file}"
    
    # ファイルの存在確認
    if not Path(video_path).exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        print("data/input/フォルダ内のファイルを確認してください。")
        exit(1)
    
    print(f"処理開始: {video_path}")
    print(f"切り出し時間: {args.time_seconds}秒")
    
    if args.interval:
        # 間隔モード: 指定間隔で複数フレーム切り出し
        print(f"間隔モード: {args.interval}秒間隔で切り出し")
        output_files = extract_frames_at_intervals(video_path, args.interval)
        if output_files:
            print(f"成功: {len(output_files)}個のフレームを切り出しました")
            for file_path in output_files:
                print(f"  - {file_path}")
        else:
            print("失敗: フレームを切り出せませんでした")
            exit(1)
    else:
        # 単一フレーム切り出しモード
        success, frame, output_path = extract_frame_at_time(
            video_path, 
            args.time_seconds, 
            args.output
        )
        
        if success:
            print(f"成功: {output_path}")
        else:
            print("失敗: フレームを切り出せませんでした")
            exit(1)