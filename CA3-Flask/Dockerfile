FROM centos:centos7
EXPOSE 5000
RUN mkdir /opt/model
WORKDIR /opt/model/
RUN yum install centos-release-scl -y
RUN yum install rh-python36 -y
RUN yum install python3-pip -y
COPY requirements.txt /opt/model/
RUN pip3 install -r requirements.txt
ENV LC_ALL=en_US.utf-8
ENV LANG=en_US.utf-8
COPY . /opt/model/
RUN ["python3", "/opt/model/model.py"]
ENTRYPOINT FLASK_APP=/opt/model/app.py flask run --host=0.0.0.0
