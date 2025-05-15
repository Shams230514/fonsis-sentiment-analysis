FROM registry.access.redhat.com/ubi8/python-39:latest

USER 0

RUN yum install -y \
    gcc \
    gcc-c++ \
    python39-devel \
    && yum clean all

USER 1001

WORKDIR /opt/app-root/src

COPY --chown=1001:0 requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=1001:0 . .

EXPOSE 8080

CMD ["python", "app.py"]
