import io
from dotenv import load_dotenv
from google.cloud import videointelligence

def vi_analysis(video_path):
    video_client = videointelligence.VideoIntelligenceServiceClient()
    
    # Use multiple features including LABEL_DETECTION and SHOT_CHANGE_DETECTION
    features = [
        videointelligence.Feature.LABEL_DETECTION,
        videointelligence.Feature.SHOT_CHANGE_DETECTION
    ]
    
    with open(video_path, 'rb') as media_file: #works perrfectly
        input_content = media_file.read()
        
    operation = video_client.annotate_video(
        request={
            'features': features,
            'input_content': input_content,
        }
    )
    
    print('\n(Temp) Processing video for label annotations and shot changes:')
    result = operation.result(timeout=180)
    print('\nFinished processing.')

    # Process segment labels
    segment_labels = result.annotation_results[0].segment_label_annotations
    for segment_label in segment_labels:
        print('Video label description: {}'.format(segment_label.entity.description))
        for i, segment in enumerate(segment_label.segments):
            start_time = (segment.segment.start_time_offset.seconds + segment.segment.start_time_offset.microseconds / 1e6)
            end_time = (segment.segment.end_time_offset.seconds + segment.segment.end_time_offset.microseconds / 1e6)
            positions = '{}s to {}s'.format(start_time, end_time)
            confidence = segment.confidence
            print('\tSegment {}: {}'.format(i, positions))
            print('\tConfidence: {}'.format(confidence))

    # Process shot change detection
    # to be re-used elsewhere
    for shot in result.annotation_results[0].shot_annotations:
        start_time = (shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6)
        end_time = (shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6)
        print('Shot change from {}s to {}s'.format(start_time, end_time))
