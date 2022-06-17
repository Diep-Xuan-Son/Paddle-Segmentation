#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"
#include "yaml-cpp/yaml.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <typeinfo>

#include <experimental/filesystem>

DEFINE_string(model_dir, "", "Directory of the inference model. "
                             "It constains deploy.yaml and infer models");
DEFINE_string(img_path, "", "Path of the test image.");
DEFINE_string(devices, "GPU", "Use GPU or CPU devices. Default: GPU");
DEFINE_bool(use_trt, false, "Wether enable TensorRT when use GPU. Defualt: false.");
DEFINE_string(trt_precision, "fp32", "The precision of TensorRT, support fp32, fp16 and int8. Default: fp32");
DEFINE_bool(use_trt_dynamic_shape, false, "Wether enable dynamic shape when use GPU and TensorRT. Defualt: false.");
DEFINE_string(dynamic_shape_path, "", "If set dynamic_shape_path, it read the dynamic shape for TRT.");
DEFINE_bool(use_mkldnn, false, "Wether enable MKLDNN when use CPU. Defualt: false.");
DEFINE_string(save_dir, "", "Directory of the output image.");

namespace fs = std::experimental::filesystem;

typedef struct YamlConfig {
  std::string model_file;
  std::string params_file;
  bool is_normalize;
  bool is_resize;
  int resize_width;
  int resize_height;
}YamlConfig;

YamlConfig load_yaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);
  std::string model_file = node["Deploy"]["model"].as<std::string>();
  std::string params_file = node["Deploy"]["params"].as<std::string>();
  YamlConfig yaml_config = {model_file, params_file};
  if (node["Deploy"]["transforms"]) {
    const YAML::Node& transforms = node["Deploy"]["transforms"];
    for (size_t i = 0; i < transforms.size(); i++) {
      if (transforms[i]["type"].as<std::string>() == "Normalize") {
        yaml_config.is_normalize = true;
      } else if (transforms[i]["type"].as<std::string>() == "Resize") {
        yaml_config.is_resize = true;
        const YAML::Node& target_size = transforms[i+1]["target_size"];
        yaml_config.resize_width = target_size[0].as<int>();
        yaml_config.resize_height = target_size[1].as<int>();
      }
    }
  }
  return yaml_config;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const YamlConfig& yaml_config) {
  std::string& model_dir = FLAGS_model_dir;

  paddle_infer::Config infer_config;
  infer_config.SetModel(model_dir + "/" + yaml_config.model_file,
                  model_dir + "/" + yaml_config.params_file);
  infer_config.EnableMemoryOptim();

  if (FLAGS_devices == "CPU") {
    LOG(INFO) << "Use CPU";
    if (FLAGS_use_mkldnn) {
      LOG(INFO) << "Use MKLDNN";
      infer_config.EnableMKLDNN();
      infer_config.SetCpuMathLibraryNumThreads(5);
    }
  } else if(FLAGS_devices == "GPU") {
    LOG(INFO) << "Use GPU";
    infer_config.EnableUseGpu(100, 0);

    // TRT config
    if (FLAGS_use_trt) {
      LOG(INFO) << "Use TRT";
      LOG(INFO) << "trt_precision:" << FLAGS_trt_precision;

      // TRT precision
      if (FLAGS_trt_precision == "fp32") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kFloat32, false, false);
      } else if (FLAGS_trt_precision == "fp16") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kHalf, false, false);
      } else if (FLAGS_trt_precision == "int8") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kInt8, false, false);
      } else {
        LOG(FATAL) << "The trt_precision should be fp32, fp16 or int8.";
      }

      // TRT dynamic shape
      if (FLAGS_use_trt_dynamic_shape) {
        LOG(INFO) << "Enable TRT dynamic shape";
        if (FLAGS_dynamic_shape_path.empty()) {
          std::map<std::string, std::vector<int>> min_input_shape = {
              {"image", {1, 3, 112, 112}}};
          std::map<std::string, std::vector<int>> max_input_shape = {
              {"image", {1, 3, 1024, 2048}}};
          std::map<std::string, std::vector<int>> opt_input_shape = {
              {"image", {1, 3, 384, 256}}};
          infer_config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                        opt_input_shape);
        } else {
          infer_config.EnableTunedTensorRtDynamicShape(FLAGS_dynamic_shape_path, true);
        }
      }
    }
  } else {
    LOG(FATAL) << "The devices should be GPU or CPU";
  }

  auto predictor = paddle_infer::CreatePredictor(infer_config);
  return predictor;
}

void hwc_img_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}
//---------------------------------------------------------------------------------------------------------
std::string find_color(int h,int s,int v){
  
  std::string color;
  if(v<17){
    color = "Black";
  }
  else if(17<=v && v<42){
    if(s>21){
      if((0<=h && h<10) || (360>=h && h>340)){
        if(s<50){color = "Brown";}
        else{color = "Red";}
      }
      else if(40>h && h>=10){
        if(s<50){color = "Brown";}
        else{color = "Orange";}
      }
      else if(68>h && h>=40){
        color = "Yellow";
      }
      else if(170>h && h>=68){
        color = "Green";
      }
      else if(260>h && h>=170){
        if(s<27){color = "Grey";}
        else{color = "Blue";}
        }
      else if(300>h && h>=260){
        color = "Purple";
      }
      else if(340>h && h>=300){
        color = "Pink";
      }
    }
    else if(5<s && s<=21){
      if(260>h && h>=170){color = "Grey";}
      else{color = "Brown";}
    }
    else{color = "Grey";}
  }
  else if(42<=v && v<60){
    if(s>16){
      if((0<=h && h<10) || (360>=h && h>340)){
        if(s<25){color = "Brown";}
        else{color = "Red";}
      }
      else if(40>h && h>=10){
        if(s<25){color = "Brown";}
        else{color = "Orange";}
      }
      else if(68>h && h>=40){
        color = "Yellow";
      }
      else if(170>h && h>=68){
        color = "Green";
      }
      else if(260>h && h>=170){
        if(s<25){color = "Grey";}
        else{color = "Blue";}
        }
      else if(300>h && h>=260){
        color = "Purple";
      }
      else if(340>h && h>=300){
        color = "Pink";
      }
    }
    else{color = "Grey";}
  }

  else if(60<=v && v<75){
    if(s>10){
      if((0<= h && h <10) || (360>= h && h >340)){
        if(s<20){color = "Brown";}
        else{color = "Red";}
      }
      else if(40>h && h>=10){
        if(s<20){color = "Brown";}
        else{color = "Orange";}
      }
      else if(68>h && h>=40){
        color = "Yellow";
      }
      else if(170>h && h>=68){
        color = "Green";
      }
      else if(260>h && h>=170){
        if(s<14){color = "Grey";}
        else{color = "Blue";}
        }
      else if(300>h && h>=260){
        color = "Purple";
      }
      else if(340>h && h>=300){
        color = "Pink";
      }
    }
    else{color = "Grey";}
  }

  else if(75<=v && v<100){
    if(s>6){
      if((0<= h && h <10) || (360>= h && h >340)){
        color = "Red";
      }
      else if(40>h && h>=10){
        color = "Orange";
      }
      else if(68>h && h>=40){
        color = "Yellow";
      }
      else if(170>h && h>=68){
        color = "Green";
      }
      else if(260>h && h>=170){
        if(s<12){color = "white";}
        else{color = "Blue";}
        }
      else if(300>h & h>=260){
        color = "Purple";
      }
      else if(340>h & h>=300){
        color = "Pink";
      }
    }
    else{color = "White";}
  }

  return color;
}
    

std::vector<int> rgb_to_hsv(float r0, float g0, float b0){
  // # R, G, B values are divided by 255
  // # to change the range from 0..255 to 0..1:
  float r = r0/255.0, g = g0/255.0, b = b0/255.0;

  // # h, s, v = hue, saturation, value
  float cmax = std::max({r, g, b}); // maximum of r, g, b
  float cmin = std::min({r, g, b}); // minimum of r, g, b
  float diff = cmax-cmin;     // diff of cmax and cmin.

  // std::cout << "diff: " << diff << std::endl;
  // # if cmax and cmax are equal then h = 0
  float h, s, v;
  if (cmax == cmin){
    h = 0;
  }
    
  // # if cmax equal r then compute h
  else if(cmax == r){
    h = fmod((60 * ((g - b) / diff) + 360), 360);
  }

  // # if cmax equal g then compute h
  else if(cmax == g){
    h = fmod((60 * ((b - r) / diff) + 120), 360.0);
  }

  // # if cmax equal b then compute h
  else if(cmax == b){
    h = fmod((60 * ((r - g) / diff) + 240), 360.0);
  }
      
  // # if cmax equal zero
  if(cmax == 0){
    s = 0;
  }
  else{
    s = (diff / cmax) * 100;
  }

  // # compute v
  v = cmax * 100;
  std::vector<int> hsv = {round(h), round(s), round(v)};
  return hsv;
}


void dominantcolor(cv::Mat im, cv::Mat result, int label){
  cv::Mat mask = result == label;
  cv::Mat im_attribute;
  im.copyTo(im_attribute,mask);

  std::vector<cv::Vec3b> vect0 {0,0,0};
  // std::cout << im_attribute << std::endl;
  // std::cout << im_attribute.at<cv::Vec3b>(383,255) == abc[0] << std::endl;

  std::vector<cv::Vec3b> vect_attribute;
  for(int i=0; i<im.rows-1; i++){
    for(int j=0; j<im.cols-1; j++){
      if(im_attribute.at<cv::Vec3b>(i,j) != vect0[0]){
        vect_attribute.push_back(im_attribute.at<cv::Vec3b>(i,j));
      }
    }
  }
  // std::cout << vect_attribute.size() << std::endl;
  cv::Mat mat_attribute(vect_attribute.size(), 1, CV_8UC3, vect_attribute.data());
  mat_attribute.convertTo(mat_attribute, CV_32F);
  // std::cout << mat_attribute.size << std::endl;
  //-----------------------------------------------------------

  // cv::Mat source_img=cv::imread(FLAGS_img_path,1); // Read image from path
  // // Serialize, float
  // cv::Mat data = source_img.reshape(1, source_img.total());
  // data.convertTo(data, CV_32F);
  
  std::string attribute;
  switch(label){
    case 1: attribute = "skin"; break;
    case 2: attribute = "bag"; break;
    case 3: attribute = "pant"; break;
    case 4: attribute = "shirt"; break;
    case 5: attribute = "shoe"; break;
    case 6: attribute = "skirt"; break;
    case 8: attribute = "hair"; break;
    case 9: attribute = "hat"; break;
  }
  // std::cout << attribute << std::endl;

  if(mat_attribute.empty()){
    std::cout << "Don't have " + attribute << std::endl;
  }
  else{
    // Perform k-Means
    int k = 3;
    std::vector<int> labels;
    cv::Mat3f centers;
    cv::kmeans(mat_attribute, k, labels, cv::TermCriteria(), 3, cv::KMEANS_PP_CENTERS,   centers);

    centers = centers.reshape(3, centers.rows);
    // data = data.reshape(3, data.rows);
    // std::cout<<"centers: " << centers <<std::endl;
    int max_count = 0;
    int max_index = 0;
    for(int i = 0; i<k; i++){
      if(max_count < std::count(labels.begin(),labels.end(),i)){
        max_count = std::count(labels.begin(),labels.end(),i);
        max_index = i;
      }
      // std::cout<<"label: " << std::count(labels.begin(),labels.end(),i) <<std::endl;
    }
    // std::cout<<"max_count: " << max_count <<std::endl;
    // std::cout<<"max_index: " << max_index <<std::endl;
    // std::cout<<"max_center: " << round((*centers[max_index])[0]) <<std::endl;
    // int r = round((*centers[max_index])[2]), g = round((*centers[max_index])[1]), b = round((*centers[max_index])[0]);
    std::vector<int> hsv = rgb_to_hsv((*centers[max_index])[2],(*centers[max_index])[1],(*centers[max_index])[0]);
    std::cout << "hsv: " << hsv.at(0) << "/" << hsv.at(1) << "/" << hsv.at(2) << std::endl;
    std::string color = find_color(hsv.at(0), hsv.at(1), hsv.at(2));
    std::cout << "Color of " + attribute +": " + color << std::endl; 
  }
}
//------------------------------------------------------------------------------------------------------------------------
cv::Mat read_process_image(const YamlConfig& yaml_config, std::string input_image) {
  cv::Mat img = cv::imread(input_image, cv::IMREAD_COLOR);
  // std::cout << "img: " << img.channels() <<std::endl;
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  if (yaml_config.is_resize) {
    cv::resize(img, img, cv::Size(yaml_config.resize_width, yaml_config.resize_height));
  }
  if (yaml_config.is_normalize) {
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    img = (img - 0.5) / 0.5;
  }
  return img;
}


int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(FATAL) << "The model_dir should not be empty.";
  }

  // Save dir
  std::string save_dir = FLAGS_save_dir;  //them
  
  // Load yaml
  std::string yaml_path = FLAGS_model_dir + "/deploy.yaml";
  YamlConfig yaml_config = load_yaml(yaml_path);

  std::string input_folder = "./data_test"; //them
  for (auto& p :  fs::recursive_directory_iterator(input_folder)){  //them
    if (p.path().extension() == ".png" || p.path().extension() == ".jpg") {
      std::string input_image = p.path().string(); //them
      cv::Mat img0 = cv::imread(input_image, cv::IMREAD_COLOR);//them
      // Prepare data
      cv::Mat img = read_process_image(yaml_config, input_image);
      // cv::imwrite("windowName.jpg", img); // Show our image inside the created window.
      int rows = img.rows;
      int cols = img.cols;
      int chs = img.channels();
      std::vector<float> input_data(1 * chs * rows * cols, 0.0f);
      hwc_img_2_chw_data(img, input_data.data());

      // Create predictor
      auto predictor = create_predictor(yaml_config);
      auto t_start = std::chrono::high_resolution_clock::now();           //them
      // Set input
      auto input_names = predictor->GetInputNames();
      auto input_t = predictor->GetInputHandle(input_names[0]);
      std::vector<int> input_shape = {1, chs, rows, cols};
      input_t->Reshape(input_shape);
      input_t->CopyFromCpu(input_data.data());
      
      // Run
      predictor->Run();

      std::cout << "==================================================" << std::endl; //them
      std::cout << input_image << std::endl; //them
      // Get output
      auto output_names = predictor->GetOutputNames();
      auto output_t = predictor->GetOutputHandle(output_names[0]);
      std::vector<int> output_shape = output_t->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());
      std::vector<int64_t> out_data(out_num);
      output_t->CopyToCpu(out_data.data());

      // Get pseudo image
      std::vector<uint8_t> out_data_u8(out_num);
      for (int i = 0; i < out_num; i++) {
        out_data_u8[i] = static_cast<uint8_t>(out_data[i]);
      }
      cv::Mat out_gray_img(output_shape[1], output_shape[2], CV_8UC1, out_data_u8.data());
      cv::Mat out_eq_img;
      // std::cout<<out_gray_img<<std::endl;  //them
      cv::equalizeHist(out_gray_img, out_eq_img);
      //-------------------------------------
      int list_attribute[] = {3,4,5,6,8}; 
      int len = sizeof(list_attribute)/sizeof(list_attribute[0]);
      for(int i=0; i<len; i++){
        dominantcolor(img0,out_gray_img,list_attribute[i]);  //them
        std::cout << "----------------------" << std::endl;
      }
      //-------------------------------------------
      std::string name_image = p.path().filename().string();//them
      cv::imwrite(save_dir + "/" + name_image, out_eq_img);
      
      LOG(INFO) << "Finish, the result is saved in out_img.jpg";

      auto t_end = std::chrono::high_resolution_clock::now();             //them
      std::cout << "Duration time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " milliseconds" << std::endl;
    }
  }
  
}
