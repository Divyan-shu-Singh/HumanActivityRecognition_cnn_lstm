import cv2
import numpy as np
from collections import deque
from pytube import YouTube
import tensorflow
from tensorflow.keras.models import load_model
from moviepy.editor import *




def download_youtube_videos(youtube_video_url, output_directory):
    '''
   This function downloads the youtube video whose URL is passed to it as an argument.
   Args:
       youtube_video_url: URL of the video that is required to be downloaded.
       output_directory:  The directory path to which the video needs to be stored after downloading.
   Returns:
       title: The title of the downloaded youtube video.
   '''

    # Create a video object which contains useful information about the video.
    video = YouTube(youtube_video_url)

    # Retrieve the title of the video.
    title = video.title

    # filters out all the files with "mp4" extension
    stream = video.streams.filter(file_extension='mp4')[0]

    # Construct the output file path.
    output_file_path = f'{output_directory}'

    # Download the youtube video at the best available quality and store it to the contructed path.
    stream.download(output_file_path)

    # Return the video title.
    return title

def predict_on_video(output_file_path, SEQUENCE_LENGTH, model, video_file_path=0):
    '''
    This function will perform action recognition on a video using the specified model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = classes_list[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)

        cv2.imshow('Predicted Frames', frame)

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord('q'):
            break

    cv2.destroyAllWindows()

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()



if __name__ == '__main__':
    classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "Lunges", "Punch", "PushUps"]

    SEQUENCE_LENGTH = 20

    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

    model1 = load_model('models/LRCN_model.h5')

    # Make the Output directory if it does not exist
    test_videos_directory = 'test_videos'
    os.makedirs(test_videos_directory, exist_ok=True)

    # Download a YouTube Video.
    # video_title = download_youtube_videos('https://www.youtube.com/watch?v=8u0qjmHIOcE', test_videos_directory)

    # Get the YouTube Video's path we just downloaded.
    # input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

    # Construct the output video path.
    output_video_file_path = f'Output-SeqLen{SEQUENCE_LENGTH}.mp4'

    # Perform Action Recognition on the Test Video.
    predict_on_video(output_video_file_path, SEQUENCE_LENGTH=20, model=model1, video_file_path=f'{test_videos_directory}/test_.mp4')

    # Display the output video.
    VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None)).ipython_display()



