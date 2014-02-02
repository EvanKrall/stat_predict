FROM lucid_yelp:latest
RUN echo 'deb http://apt.local.yelpcorp.com/ubuntu/custom/ lucid-yelp soabase' > /etc/apt/sources.list.d/yelppack-soabase.list && \
    echo 'deb http://apt.local.yelpcorp.com/ubuntu/custom/ lucid-yelp optional' > /etc/apt/sources.list.d/yelppack-optional.list && \
    echo 'deb http://apt.local.yelpcorp.com/ubuntu/custom/ lucid-yelp dev' > /etc/apt/sources.list.d/yelppack-dev.list && \
    apt-get update

RUN apt-get install -y python-numpy python2.6-dev python-pip python-yaml rrdtool python-rrdtool
RUN pip install docopt
RUN mkdir /workdir
WORKDIR /workdir

RUN apt-get install -y python-lxml
RUN apt-get install -y python-matplotlib
