FROM registry.cn-shenzhen.aliyuncs.com/hukefei/eggtart:1.1

# Install mmdetection
RUN rm -r /models
ADD . /
WORKDIR /

CMD sh run.sh