#!/usr/bin/env python

from pytube import YouTube
from mas_framework.utils.const import mas_framework_ROOT

def Youtube(url, has_subtitles):
    video_id=url.split('v=')[-1].split('&')[0]
    youtube = YouTube(url)
    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if has_subtitles:
        print('Downloading video')
        video_stream.download(output_path="{GDesigner_ROOT}/workspace",filename=f"{video_id}.mp4")
        print('Video downloaded successfully')
        return f"{mas_framework_ROOT}/workspace/{video_id}.mp4"
    else:
        return video_stream.url 