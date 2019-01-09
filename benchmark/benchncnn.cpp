#include <float.h>
#include <stdio.h>
#include <algorithm>

#ifdef _WIN32
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include "benchmark.h"
#include "cpu.h"
#include "net.h"

namespace ncnn {

// always return empty weights
class ModelBinFromEmpty : public ModelBin
{
public:
    virtual Mat load(int w, int /*type*/) const { return Mat(w); }
};

class BenchNet : public Net
{
public:
    int load_model()
    {
        // load file
        int ret = 0;

        ModelBinFromEmpty mb;
        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];

            int lret = layer->load_model(mb);
            if (lret != 0)
            {
                fprintf(stderr, "layer load_model %d failed\n", (int)i);
                ret = -1;
                break;
            }
        }

        return ret;
    }
};

} // namespace ncnn

static int g_loop_count = 4;

void benchmark(const char* comment, void (*init)(ncnn::Net&), void (*run)(const ncnn::Net&), bool load_model_from_out = false)
{
    ncnn::BenchNet net;

    init(net);

	if(!load_model_from_out)
		net.load_model();

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
    sleep(10);
#endif

    // warm up
    run(net);
    run(net);
    run(net);

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i=0; i<g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        run(net);

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = (std::min)(time_min, time);
        time_max = (std::max)(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%16s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

void myNet_init(ncnn::Net& net)
{
	net.load_param("MNet18_v2_3.param");
	net.load_model("MNet18_v2_3.bin");
}

void myNet_run(const ncnn::Net& net)
{
	ncnn::Extractor ex = net.create_extractor();

	ncnn::Mat in(256, 320, 3);
	ex.input("data", in);

	ncnn::Mat out1, out2, out3, out4, out5, out6, out7, out8;
	ex.extract("detect_6_2_cls_softmax_6_2", out1);
	ex.extract("detect_6_2_reg", out2);
	ex.extract("detect_fc7_cls_softmax_fc7", out3);
	ex.extract("detect_fc7_reg", out4);
	ex.extract("detect_5_3_cls_softmax_5_3", out5);
	ex.extract("detect_5_3_reg", out6);
	ex.extract("detect_4_3_cls_softmax_4_3", out7);
	ex.extract("detect_4_3_reg", out8);
}

void peleenet_init(ncnn::Net& net)
{
	net.load_param("pelee.param");
}

void peleenet_run(const ncnn::Net& net)
{
	ncnn::Extractor ex = net.create_extractor();

	ncnn::Mat in(304, 304, 3);
	ex.input("data", in);

	ncnn::Mat out;
	ex.extract("detection_out", out);
}

void faceboxnet_init(ncnn::Net& net)
{
	net.load_param("faceboxes.param");
}

void faceboxnet_run(const ncnn::Net& net)
{
	ncnn::Extractor ex = net.create_extractor();

	ncnn::Mat in(1024, 1024, 3);
	ex.input("data", in);

	ncnn::Mat out;
	ex.extract("detection_out", out);
}

void squeezenet_init(ncnn::Net& net)
{
    net.load_param("squeezenet_v1.1.param");
}

void squeezenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_init(ncnn::Net& net)
{
    net.load_param("mobilenet.param");
}

void mobilenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_v2_init(ncnn::Net& net)
{
    net.load_param("mobilenet_v2.param");
}

void mobilenet_v2_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(320, 256, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void shufflenet_init(ncnn::Net& net)
{
    net.load_param("shufflenet.param");
}

void shufflenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1000", out);
}

void googlenet_init(ncnn::Net& net)
{
    net.load_param("googlenet.param");
}

void googlenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void resnet18_init(ncnn::Net& net)
{
    net.load_param("resnet18.param");
}

void resnet18_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void alexnet_init(ncnn::Net& net)
{
    net.load_param("alexnet.param");
}

void alexnet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void vgg16_init(ncnn::Net& net)
{
    net.load_param("vgg16.param");
}

void vgg16_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void squeezenet_ssd_init(ncnn::Net& net)
{
    net.load_param("squeezenet_ssd.param");
}

void squeezenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(300, 300, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

void mobilenet_ssd_init(ncnn::Net& net)
{
    net.load_param("mobilenet_ssd.param");
}

void mobilenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(300, 300, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }

    g_loop_count = loop_count;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());

    // run
	benchmark("myNet", myNet_init, myNet_run, true);

    benchmark("squeezenet_v1.1", squeezenet_init, squeezenet_run);

	benchmark("mobilenet_v2", mobilenet_v2_init, mobilenet_v2_run);

	benchmark("Pelee", peleenet_init, peleenet_run);

	benchmark("FaceBox", faceboxnet_init, faceboxnet_run);

    /*benchmark("mobilenet", mobilenet_init, mobilenet_run);

    benchmark("shufflenet", shufflenet_init, shufflenet_run);

    benchmark("googlenet", googlenet_init, googlenet_run);

    benchmark("resnet18", resnet18_init, resnet18_run);

    benchmark("alexnet", alexnet_init, alexnet_run);

    benchmark("vgg16", vgg16_init, vgg16_run);

    benchmark("squeezenet-ssd", squeezenet_ssd_init, squeezenet_ssd_run);

    benchmark("mobilenet-ssd", mobilenet_ssd_init, mobilenet_ssd_run);*/

    return 0;
}
