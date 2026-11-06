import json
import os

def convert_ground_truth_to_json(txt_file_path, json_file_path):
    """
    将ground_truth.txt转换为JSON格式
    
    输入格式:
    [image_filename] [valence] [arousal] [prob. anger] [prob. disgust] [prob. fear] [prob. joy] [prob. sadness] [prob. surprise] [prob. neutral]
    
    输出JSON格式:
    {
        "image_filename": {
            "valence": float,
            "arousal": float,
            "emotions": {
                "anger": float,
                "disgust": float,
                "fear": float,
                "joy": float,
                "sadness": float,
                "surprise": float,
                "neutral": float
            },
            "dominant_emotion": str,
            "emotion_label": int
        }
    }
    """
    
    # 情绪类别映射（按字母顺序）
    emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotion_classes)}
    
    data = {}
    
    with open(txt_file_path , 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过第一行（标题行）
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) != 10:
            continue
            
        image_filename = parts[0]
        valence = float(parts[1])
        arousal = float(parts[2])
        
        # 情绪概率
        emotions = {
            'anger': float(parts[3]),
            'disgust': float(parts[4]),
            'fear': float(parts[5]),
            'joy': float(parts[6]),
            'sadness': float(parts[7]),
            'surprise': float(parts[8]),
            'neutral': float(parts[9])
        }
        
        # 找出主导情绪（概率最高的情绪）
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        # 从文件名中提取真实标签
        true_emotion = image_filename.split('/')[0]
        
        # 创建数据条目
        data[image_filename] = {
            'valence': valence,
            'arousal': arousal,
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'true_emotion': true_emotion,
            'emotion_label': emotion_to_label.get(true_emotion, -1),
            'dominant_emotion_label': emotion_to_label.get(dominant_emotion, -1)
        }
    
    # 保存为JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"Successfully converted {len(data)} samples")
    
    # 统计每个类别的样本数
    class_counts = {}
    for item in data.values():
        emotion = item['true_emotion']
        class_counts[emotion] = class_counts.get(emotion, 0) + 1
    
    print("\nSamples per class:")
    for emotion, count in sorted(class_counts.items()):
        print(f"  {emotion}: {count}")
    
    # 统计预测准确率（基于概率最高的情绪）
    correct = sum(1 for item in data.values() if item['true_emotion'] == item['dominant_emotion'])
    accuracy = correct / len(data) * 100
    print(f"\nAccuracy based on probabilities: {accuracy:.2f}%")
    
    return data

# 使用示例
if __name__ == "__main__":
    # 转换文件
    ground_truth_data = convert_ground_truth_to_json('ground_truth.txt', 'emotion6_ground_truth.json')
    
