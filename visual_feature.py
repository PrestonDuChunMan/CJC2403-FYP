import cv2

def analyze_visual_tempo(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    motion_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for easier analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # Calculate the absolute difference between frames
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels (indicating motion)
        motion = cv2.countNonZero(thresh)
        if motion > 1000:  # Threshold for detecting significant motion
            motion_frames += 1

        prev_frame = gray

    cap.release()
    print(f"Motion frames detected: {motion_frames}")

# Example usage
analyze_visual_tempo('./greenery.mp4')