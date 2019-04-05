TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    ELM_CNN_Model.cpp \
    functions.cpp

INCLUDEPATH += usr/local/include
LIBS += /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_imgcodecs.so

HEADERS += \
    ELM_CNN_Model.h \
    functions.h
