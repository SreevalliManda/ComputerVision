import argparse
import random
import time
import cv2

# Initialize Selective Search
def selective_search(image, method="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    
    if method == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    
    return ss

# Main processing function
def process_image(image_path, method="fast"):
    # Load input image
    image = cv2.imread(image_path)
    
    # Initialize Selective Search
    ss = selective_search(image, method)
    
    # Process Selective Search
    start = time.time()
    rects = ss.process()
    end = time.time()
    
    print(f"Selective Search took {end - start:.4f} seconds")
    print(f"Found {len(rects)} region proposals")
    
    return image, rects

# Visualization function
def visualize_proposals(image, rects, step=100):
    for i in range(0, len(rects), step):
        clone = image.copy()
        for (x, y, w, h) in rects[i:i+step]:
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(clone, (x, y), (x+w, y+h), color, 2)
        
        cv2.imshow("Region Proposals", clone)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-m", "--method", choices=["fast", "quality"], 
                        default="fast", help="Selective search method")
    args = parser.parse_args()

    image, rects = process_image(args.image, args.method)
    visualize_proposals(image, rects)
    cv2.destroyAllWindows()


# python 2_selectivesearch.py --image input.jpg --method fast
