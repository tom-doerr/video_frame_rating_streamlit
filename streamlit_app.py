import streamlit as st
import cv2
import time
import os
from clip_client import Client
from docarray import Document
import PIL

st.set_page_config(page_title='AI Video Frame Rater', initial_sidebar_state="auto")

st.title('AI Video Frame Rater')

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mkv"])

c = Client('grpcs://demo-cas.jina.ai:2096')

if not uploaded_video:
    st.stop()

FRAMES_DIR = 'frames'
os.makedirs(FRAMES_DIR, exist_ok=True)

def video_to_frames(video_file):

    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        cv2.imwrite(f'{FRAMES_DIR}/frame{count}.jpg', image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        images.append(image)

    return images


def video_to_frames_generator(video_file):
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        while success:
            # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
            cv2.imwrite(f'{FRAMES_DIR}/frame{count}.jpg', image)  # save frame as JPEG file
            success, image = vidcap.read()
            print(f'Read frame {count}: ', success)
            count += 1
            yield image, success




# save the video
video_filename = 'video.mp4'
with open(video_filename, 'wb') as f:
    f.write(uploaded_video.getbuffer())



def rate_image(image_path, target, opposite, attempt=0):
    try:
        r = c.rank(
            [
                Document(
                    # uri='https://www.pngall.com/wp-content/uploads/12/Britney-Spears-PNG-Image-File.png',
                    uri=image_path,
                    matches=[
                        Document(text=target),
                        Document(text=opposite),
                    ],
                )
            ]
        )
    except (ConnectionError, AioRpcError) as e:
        print(e)
        print(f'Retrying... {attempt}')
        time.sleep(2**attempt)
        return rate_image(image_path, target, opposite, attempt + 1)
    text_and_scores = r['@m', ['text', 'scores__clip_score__value']]
    index_of_good_text = text_and_scores[0].index(target)
    score =  text_and_scores[1][index_of_good_text]
    return score

def process_image(photo_file, metrics):  
    col1, col2, col3 = st.columns([10,10,10])
    # with st.spinner('Loading...'):
        # with col1:
            # st.write('')
        # with col2:
            # st.image(photo_file, use_column_width=True)
        # with col3:
            # st.write('')


    # save it
    filename = f'{time.time()}'.replace('.', '_')
    filename_path = f'{IMAGES_FOLDER}/{filename}'
    # save the numpy image to a file
    with open(f'{filename_path}', 'wb') as f:
        image_data_converted = cv2.imencode('.jpg', photo_file)[1].tobytes()
        f.write(image_data_converted)





    # with st.spinner('Rating your photo...'):
    scores = dict()
    for metric in metrics:
        target = metric_texts[metric][0]
        opposite = metric_texts[metric][1]
        score = rate_image(filename_path, target, opposite)
        scores[metric] = score


    scores['Avg'] = sum(scores.values()) / len(scores)

        # plot them

    return filename_path, scores




def plot_metrics(metrics):
    st.title('Metrics')
    import plotly.graph_objects as go


    scores_percent = []
    for metric in metrics:
        scores_percent.append(scores[metric] * 100)
    fig = go.Figure(data=[go.Bar(x=metrics, y=scores_percent)], layout=go.Layout(title='Scores'))
    # range 0 to 100 for the y axis:
    fig.update_layout(yaxis=dict(range=[0, 100]))

    st.plotly_chart(fig, use_container_width=True)


IMAGES_FOLDER = 'images'
PAGE_LOAD_LOG_FILE = 'page_load_log.txt'
METRIC_TEXTS = {
    'Attractivness': ('this person is attractive', 'this person is unattractive'),
    'Hotness': ('this person is hot', 'this person is ugly'),
    'Trustworthiness': ('this person is trustworthy', 'this person is dishonest'),
    'Intelligence': ('this person is smart', 'this person is stupid'),
    'Quality': ('this image looks good', 'this image looks bad'),
}




def log_page_load():
    with open(PAGE_LOAD_LOG_FILE, 'a') as f:
        f.write(f'{time.time()}\n')


def get_num_page_loads():
    with open(PAGE_LOAD_LOG_FILE, 'r') as f:
        return len(f.readlines())

def get_earliest_page_load_time():
    with open(PAGE_LOAD_LOG_FILE, 'r') as f:
        lines = f.readlines()
        unix_time = float(lines[0])

    date_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(unix_time))
    return date_string



def show_sidebar_metrics():
    metric_options = list(METRIC_TEXTS.keys())
    # default_metrics = ['Attractivness', 'Trustworthiness', 'Intelligence'] 
    default_metrics = ['Hotness']
    st.sidebar.title('Metrics')
    # metric = st.sidebar.selectbox('Select a metric', metric_options)
    selected_metrics = []
    for metric in metric_options:
        selected = metric in default_metrics
        if st.sidebar.checkbox(metric, selected):
            selected_metrics.append(metric)

    with st.sidebar.expander('Metric texts'):
        st.write(METRIC_TEXTS)

    print("selected_metrics:", selected_metrics)
    return selected_metrics


def get_custom_metric():
    st.sidebar.markdown('**Custom metric**:')
    metric_name = st.sidebar.text_input('Metric name', placeholder='e.g. "Youth"')
    metric_target = st.sidebar.text_input('Metric target', placeholder='this person is young')
    metric_opposite = st.sidebar.text_input('Metric opposite', placeholder='this person is old')
    return {metric_name: (metric_target, metric_opposite)}




log_page_load()

step_size = st.sidebar.number_input('Step size', value=10, min_value=1)
number_best_images_to_show = st.sidebar.number_input('Number of best images to show', value=100, min_value=1)
# fps = st.sidebar.number_input('FPS', value=10)
st.sidebar.markdown('---')
metrics = show_sidebar_metrics()
st.sidebar.markdown('---')
custom_metric = get_custom_metric()
st.sidebar.markdown('---')
st.sidebar.write(f'Page loads: {get_num_page_loads()}')
st.sidebar.write(f'Earliest page load: {get_earliest_page_load_time()}')

metric_texts = METRIC_TEXTS
print("custom_metric:", custom_metric)
custom_key = list(custom_metric.keys())[0]
if custom_key:
    custom_tuple = custom_metric[custom_key]
    if custom_tuple[0] and custom_tuple[1]:
        metrics.append(list(custom_metric.keys())[0])
        metric_texts = {**metric_texts, **custom_metric}



os.makedirs(IMAGES_FOLDER, exist_ok=True)


if len(metrics) == 0:
    st.write('No metrics selected')
    st.stop()








# images = video_to_frames(video_filename)
frames_generator = video_to_frames_generator(video_filename)
current_frame_score_field = st.empty()
current_image = st.empty()
best_frame_score_field = st.empty()
best_image = st.empty()
filename_scores = []
time_last_frame = 0
for i, (image, success_loading_image) in enumerate(frames_generator):
    if not success_loading_image:
        st.write('Done processing video')
        break

    if i % step_size == 0:
        filename_path, scores = process_image(image, metrics)
        score = scores['Avg']
        current_frame_score_field.write(f'Current frame (frame {i}, {score:.3f}):')
        filename_scores.append((filename_path, scores))
        filename_scores.sort(key=lambda x: x[1]['Avg'], reverse=True)
        best_score = filename_scores[0][1]['Avg']
        best_frame_score_field.write(f'Best frame (score {best_score:.3f}):')
        best_image.image(filename_scores[0][0])


    # if (time.time() - time_last_frame) > 1 / fps or i % step_size == 0:
        image_other_colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        current_image.image(image_other_colors)
        time_last_frame = time.time()


# show the best images
st.title('Best images')
for filename_path, scores in filename_scores[:number_best_images_to_show]:
    st.image(filename_path, use_column_width=True, caption=f'Score: {scores["Avg"]:.3f}')

