FROM registry.cn-shenzhen.aliyuncs.com/eggtart/taobaolive:1.0

ADD . /
WORKDIR /

CMD sh run.sh