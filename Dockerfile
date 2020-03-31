FROM registry.cn-shenzhen.aliyuncs.com/eggtart/taobaolive:1.0

RUN rm -r /taobao_models
ADD . /
WORKDIR /

CMD sh run.sh