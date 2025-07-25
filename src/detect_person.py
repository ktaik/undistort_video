"""
画像から人物を検出して座標を返すモジュール
"""
import os
import sys

# PYTHONPATHとROSの影響を排除
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# sys.pathからROSとローカルライブラリを除去
original_paths = sys.path.copy()
sys.path = [p for p in sys.path if not any([
    p.startswith('/opt/ros'),
    p.startswith('/home/ktaik/.local/lib/python3.8'),
    'dist-packages' in p and 'python3.8' in p
])]

import cv2
import numpy as np
import argparse
from pathlib import Path
import torch

class PersonDetector:
    def __init__(self, model_type='yolov8'):
        """
        人物検出器の初期化
        
        Args:
            model_type (str): 使用するモデル ('yolov5', 'yolov7', 'yolov8', 'yolo-nas')
        """
        self.model_type = model_type
        
        # YOLO系モデル
        self.yolo_model = None
        
        # CUDAの利用可能性を安全にチェック
        try:
            import torch
            self.device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
        except (ImportError, AttributeError):
            self.device = 'cpu'
            
        print(f"使用デバイス: {self.device}")
        
        # モデルを初期化
        self._initialize_model()
        
    def _initialize_model(self):
        """使用するモデルを初期化"""
        try:
            if self.model_type == 'yolov5':
                # YOLOv5
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.to(self.device)
                print(f"YOLOv5モデルを初期化しました（デバイス: {self.device}）")
                
            elif self.model_type == 'yolov7':
                # YOLOv7
                try:
                    import yolov7
                    self.yolo_model = yolov7.load('yolov7.pt', device=self.device)
                    print(f"YOLOv7モデルを初期化しました（デバイス: {self.device}）")
                except ImportError:
                    print("YOLOv7が利用できません。pip install yolov7-package を実行してください")
                    print("YOLOv8にフォールバックします")
                    self.model_type = 'yolov8'
                    self._initialize_yolov8()
                    
            elif self.model_type == 'yolov8':
                self._initialize_yolov8()
                    
            elif self.model_type == 'yolo-nas':
                # YOLO-NAS
                try:
                    from super_gradients.training import models
                    self.yolo_model = models.get('yolo_nas_s', pretrained_weights="coco")
                    self.yolo_model.to(self.device)
                    print(f"YOLO-NASモデルを初期化しました（デバイス: {self.device}）")
                except ImportError:
                    print("YOLO-NASが利用できません。pip install super-gradients を実行してください")
                    print("YOLOv8にフォールバックします")
                    self.model_type = 'yolov8'
                    self._initialize_yolov8()
                
        except Exception as e:
            print(f"モデル初期化エラー: {e}")
            print("YOLOv8にフォールバックします")
            self.model_type = 'yolov8'
            self._initialize_yolov8()
    
    def _initialize_yolov8(self):
        """YOLOv8を初期化"""
        try:
            import sys
            print(f"Python version: {sys.version}")
            print(f"Python path: {sys.path[:3]}...")  # 最初の3つのパスのみ表示
            
            from ultralytics import YOLO
            print("ultralytics import 成功")
            
            self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
            self.device = 'cpu'  # CPUのみを使用
            print(f"YOLOv8モデルを初期化しました（デバイス: {self.device}）")
            
        except ImportError as e:
            print(f"ImportError詳細: {e}")
            print("ultralyticsのインポートに失敗しました")
            
            # インストール状況を確認
            try:
                import subprocess
                result = subprocess.run(['pip', 'show', 'ultralytics'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("ultralyticsはインストールされています:")
                    print(result.stdout)
                else:
                    print("ultralyticsがインストールされていません")
            except Exception as check_e:
                print(f"インストール確認エラー: {check_e}")
                
            raise RuntimeError("YOLOv8が利用できません。仮想環境でpip install ultralytics を実行してください")
            
        except Exception as e:
            print(f"YOLOv8初期化エラー: {e}")
            print(f"エラータイプ: {type(e)}")
            raise RuntimeError(f"YOLOv8の初期化に失敗しました: {e}")
        
    def detect_persons_yolo(self, image_path):
        """
        YOLO系モデルを使用して人物を検出
        
        Args:
            image_path (str): 画像ファイルパス
            
        Returns:
            list: 検出された人物の座標リスト
        """
        if self.yolo_model is None:
            raise RuntimeError("YOLOモデルが初期化されていません")
        
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"エラー: 画像を読み込めませんでした: {image_path}")
            return []
        
        try:
            if self.model_type in ['yolov5', 'yolov7']:
                # YOLOv5/v7の推論
                results = self.yolo_model(image)
                
                # 結果を解析
                detections = []
                for result in results.pandas().xyxy[0].itertuples():
                    # person class (class 0 in COCO)
                    if result.class_ == 0 and result.confidence > 0.5:
                        x1, y1, x2, y2 = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax)
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            'x': x1,
                            'y': y1,
                            'width': w,
                            'height': h,
                            'confidence': float(result.confidence),
                            'center_x': int(x1 + w/2),
                            'center_y': int(y1 + h/2)
                        })
                        
            elif self.model_type == 'yolov8':
                # YOLOv8の推論
                results = self.yolo_model(image)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # person class (class 0 in COCO)
                            if int(box.cls) == 0 and float(box.conf) > 0.5:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                w, h = x2 - x1, y2 - y1
                                
                                detections.append({
                                    'x': x1,
                                    'y': y1,
                                    'width': w,
                                    'height': h,
                                    'confidence': float(box.conf),
                                    'center_x': int(x1 + w/2),
                                    'center_y': int(y1 + h/2)
                                })
                                
            elif self.model_type == 'yolo-nas':
                # YOLO-NASの推論
                results = self.yolo_model.predict(image)
                
                detections = []
                for result in results:
                    boxes = result.prediction.bboxes_xyxy
                    labels = result.prediction.labels
                    scores = result.prediction.confidence
                    
                    for box, label, score in zip(boxes, labels, scores):
                        # person class (class 0 in COCO)
                        if int(label) == 0 and float(score) > 0.5:
                            x1, y1, x2, y2 = box
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1
                            
                            detections.append({
                                'x': x1,
                                'y': y1,
                                'width': w,
                                'height': h,
                                'confidence': float(score),
                                'center_x': int(x1 + w/2),
                                'center_y': int(y1 + h/2)
                            })
            
            return detections
            
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            raise RuntimeError(f"人物検出に失敗しました: {e}")
    
    def detect_persons_yolo_from_array(self, image_array):
        """
        numpy配列からYOLO系モデルで人物を検出（mainから呼び出し用）
        
        Args:
            image_array (numpy.ndarray): OpenCVで読み込んだ画像配列
            
        Returns:
            list: 検出された人物の座標リスト
        """
        if self.yolo_model is None:
            raise RuntimeError("YOLOモデルが初期化されていません")
        
        if image_array is None:
            return []
        
        try:
            if self.model_type in ['yolov5', 'yolov7']:
                # YOLOv5/v7の推論
                results = self.yolo_model(image_array)
                
                detections = []
                for result in results.pandas().xyxy[0].itertuples():
                    if result.class_ == 0 and result.confidence > 0.5:
                        x1, y1, x2, y2 = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax)
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            'x': x1,
                            'y': y1,
                            'width': w,
                            'height': h,
                            'confidence': float(result.confidence),
                            'center_x': int(x1 + w/2),
                            'center_y': int(y1 + h/2)
                        })
                        
            elif self.model_type == 'yolov8':
                # YOLOv8の推論
                results = self.yolo_model(image_array)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if int(box.cls) == 0 and float(box.conf) > 0.5:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                w, h = x2 - x1, y2 - y1
                                
                                detections.append({
                                    'x': x1,
                                    'y': y1,
                                    'width': w,
                                    'height': h,
                                    'confidence': float(box.conf),
                                    'center_x': int(x1 + w/2),
                                    'center_y': int(y1 + h/2)
                                })
                                
            elif self.model_type == 'yolo-nas':
                # YOLO-NASの推論
                results = self.yolo_model.predict(image_array)
                
                detections = []
                for result in results:
                    boxes = result.prediction.bboxes_xyxy
                    labels = result.prediction.labels
                    scores = result.prediction.confidence
                    
                    for box, label, score in zip(boxes, labels, scores):
                        if int(label) == 0 and float(score) > 0.5:
                            x1, y1, x2, y2 = box
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1
                            
                            detections.append({
                                'x': x1,
                                'y': y1,
                                'width': w,
                                'height': h,
                                'confidence': float(score),
                                'center_x': int(x1 + w/2),
                                'center_y': int(y1 + h/2)
                            })
            
            return detections
            
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            raise RuntimeError(f"人物検出に失敗しました: {e}")
    
    def detect_persons(self, image_path):
        """
        設定されたモデルを使用して人物を検出
        
        Args:
            image_path (str): 画像ファイルパス
            
        Returns:
            list: 検出された人物の座標リスト
        """
        return self.detect_persons_yolo(image_path)
    
    def detect_persons_from_array(self, image_array):
        """
        numpy配列から人物を検出（mainから呼び出し用）
        
        Args:
            image_array (numpy.ndarray): OpenCVで読み込んだ画像配列
            
        Returns:
            list: 検出された人物の座標リスト
        """
        return self.detect_persons_yolo_from_array(image_array)
        """
        HOG検出器を使用して人物を検出
        
        Args:
            image_path (str): 画像ファイルパス
            
        Returns:
            list: 検出された人物の座標リスト [(x, y, w, h), ...]
        """
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"エラー: 画像を読み込めませんでした: {image_path}")
            return []
        
        # 人物検出
        boxes, weights = self.hog.detectMultiScale(
            image, 
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05
        )
        
        # 信頼度フィルタリング（weights > 0.5）
        filtered_boxes = []
        for i, weight in enumerate(weights):
            if weight > 0.5:
                x, y, w, h = boxes[i]
                filtered_boxes.append({
                    'x': int(x),
                    'y': int(y), 
                    'width': int(w),
                    'height': int(h),
                    'confidence': float(weight[0]),
                    'center_x': int(x + w/2),
                    'center_y': int(y + h/2)
                })
        
        return filtered_boxes
    
    def detect_persons_from_array(self, image_array):
        """
        numpy配列から人物を検出（mainから呼び出し用）
        
        Args:
            image_array (numpy.ndarray): OpenCVで読み込んだ画像配列
            
        Returns:
            list: 検出された人物の座標リスト
        """
        if image_array is None:
            return []
        
        # 人物検出
        boxes, weights = self.hog.detectMultiScale(
            image_array, 
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05
        )
        
        # 信頼度フィルタリング
        filtered_boxes = []
        for i, weight in enumerate(weights):
            if weight > 0.5:
                x, y, w, h = boxes[i]
                filtered_boxes.append({
                    'x': int(x),
                    'y': int(y), 
                    'width': int(w),
                    'height': int(h),
                    'confidence': float(weight[0]),
                    'center_x': int(x + w/2),
                    'center_y': int(y + h/2)
                })
        
        return filtered_boxes
    
    def draw_detections(self, image_path, output_path, detections):
        """
        検出結果を画像に描画
        
        Args:
            image_path (str): 入力画像パス
            output_path (str): 出力画像パス
            detections (list): 検出結果のリスト
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"エラー: 画像を読み込めませんでした: {image_path}")
            return False
        
        # 検出結果を描画
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            confidence = detection['confidence']
            
            # 矩形を描画
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 中心点を描画
            center_x, center_y = detection['center_x'], detection['center_y']
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 信頼度を表示
            label = f"Person: {confidence:.2f}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 出力ディレクトリを作成
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 画像を保存
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"検出結果画像を保存: {output_path}")
            return True
        else:
            print(f"エラー: 画像の保存に失敗しました: {output_path}")
            return False
    
    def get_person_regions(self, image_path, margin=50):
        """
        人物が写っている領域を取得（ゆがみ補正用）
        
        Args:
            image_path (str): 画像ファイルパス
            margin (int): 人物領域の周囲のマージン（ピクセル）
            
        Returns:
            list: 人物領域のリスト [{'region': (x, y, w, h), 'center': (cx, cy)}, ...]
        """
        detections = self.detect_persons(image_path)
        
        if not detections:
            return []
        
        # 画像サイズを取得
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        regions = []
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            
            # マージンを考慮した領域を計算
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(width, x + w + margin)
            y_end = min(height, y + h + margin)
            
            region_width = x_end - x_start
            region_height = y_end - y_start
            
            regions.append({
                'original_detection': detection,
                'region': (x_start, y_start, region_width, region_height),
                'center': (detection['center_x'], detection['center_y']),
                'confidence': detection['confidence']
            })
        
        return regions

def detect_persons(image_path, model_type='yolov8'):
    """
    便利関数：画像から人物を検出
    
    Args:
        image_path (str): 画像ファイルパス
        model_type (str): 使用するモデル
        
    Returns:
        list: 検出された人物の座標リスト
    """
    detector = PersonDetector(model_type=model_type)
    return detector.detect_persons(image_path)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='画像から人物を検出（YOLO系モデルのみ）')
    parser.add_argument('image_file', help='画像ファイル名（data/output/内のファイル）')
    parser.add_argument('--model', '-m', choices=['yolov5', 'yolov7', 'yolov8', 'yolo-nas'], 
                       default='yolov8', help='使用する検出モデル')
    parser.add_argument('--output', '-o', help='検出結果画像の出力パス（オプション）')
    parser.add_argument('--margin', type=int, default=50, help='人物領域のマージン（ピクセル）')
    parser.add_argument('--show-regions', action='store_true', help='人物領域情報を表示')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='検出の信頼度閾値')
    
    args = parser.parse_args()
    
    # 画像ファイルのフルパスを構築
    image_path = f"../data/output/{args.image_file}"
    
    # ファイルの存在確認
    if not Path(image_path).exists():
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        print("data/output/フォルダ内のファイルを確認してください。")
        exit(1)
    
    print(f"処理開始: {image_path}")
    print(f"使用モデル: {args.model}")
    
    try:
        # 人物検出器を初期化
        detector = PersonDetector(model_type=args.model)
        
        # 人物検出を実行
        detections = detector.detect_persons(image_path)
        
        if detections:
            print(f"検出された人数: {len(detections)} (モデル: {detector.model_type})")
            
            # 検出結果を表示
            for i, detection in enumerate(detections):
                print(f"人物 {i+1}:")
                print(f"  座標: ({detection['x']}, {detection['y']})")
                print(f"  サイズ: {detection['width']} x {detection['height']}")
                print(f"  中心: ({detection['center_x']}, {detection['center_y']})")
                print(f"  信頼度: {detection['confidence']:.3f}")
            
            # 検出結果画像を保存
            if args.output:
                output_path = args.output
            else:
                input_name = Path(image_path).stem
                output_path = f"../data/output/detected_{detector.model_type}_{input_name}.jpg"
            
            detector.draw_detections(image_path, output_path, detections)
            
            # 人物領域情報を表示
            if args.show_regions:
                regions = detector.get_person_regions(image_path, args.margin)
                print(f"\n=== 人物領域情報（ゆがみ補正用）- {detector.model_type} ===")
                for i, region in enumerate(regions):
                    x, y, w, h = region['region']
                    cx, cy = region['center']
                    print(f"領域 {i+1}: ({x}, {y}, {w}, {h}), 中心: ({cx}, {cy})")
        
        else:
            print(f"人物が検出されませんでした (モデル: {detector.model_type})")
            exit(1)
            
    except RuntimeError as e:
        print(f"エラー: {e}")
        exit(1)
    except Exception as e:
        print(f"予期しないエラー: {e}")
        exit(1)