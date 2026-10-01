#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <array>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct Target {
    cv::Rect box;
    float conf = 0.0f;
    bool active = false;
    int lost = 0;
};

int detect_alcor_cam_id() {
    std::array<char, 256> buffer;
    std::string result = "";
    FILE* pipe = popen("v4l2-ctl --list-devices 2>/dev/null", "r");
    if (pipe) {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
        pclose(pipe);
    }

    std::istringstream stream(result);
    std::string line;
    bool found_alcor = false;
    while (std::getline(stream, line)) {
        std::string lower_line = line;
        for (auto &c : lower_line) c = tolower(c);
        if (lower_line.find("alcor") != std::string::npos) {
            found_alcor = true;
            continue;
        }
        if (found_alcor && line.find("/dev/video") != std::string::npos) {
            size_t pos = line.find("/dev/video");
            try {
                return std::stoi(line.substr(pos + 10));
            } catch (...) {}
        }
        if (found_alcor && line.empty()) {
            found_alcor = false;
        }
    }
    return 2;
}

int main() {
    int cam_id = detect_alcor_cam_id();
    std::cout << "[OK] Kamera Belirlendi: /dev/video" << cam_id << "\n";

    std::string xml_path = "runs/detect/dart_finetuned/weights/best_openvino_model/best.xml";
    std::cout << "[OK] OpenVINO Modeli: " << xml_path << "\n";

    // 1. OpenVINO Çekirdeği
    ov::Core core;
    core.set_property("CPU", ov::inference_num_threads(8));
    core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    std::shared_ptr<ov::Model> model = core.read_model(xml_path);

    // Görüntü Dönüştürücü: BGR u8 -> RGB f32 normalize
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255.0f, 255.0f, 255.0f});
    ppp.input().model().set_layout("NCHW");
    model = ppp.build();

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 2. Kamera Kurulumu
    cv::VideoCapture cap(cam_id, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) {
        std::cerr << "HATA: Kamera açılamadı!\n";
        return 1;
    }

    cv::namedWindow("Nerf Interceptor Native OpenVINO", cv::WINDOW_AUTOSIZE);

    const float CONF_ACQUIRE = 0.40f;
    const float CONF_HOLD    = 0.22f;
    const int MAX_LOST       = 4;

    Target target;
    cv::Mat frame;
    cv::Mat letterbox = cv::Mat::zeros(640, 640, CV_8UC3);
    const int y_off = (640 - 480) / 2;

    auto t_prev = std::chrono::steady_clock::now();
    float fps = 0.0f;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            continue;
        }

        int orig_w = frame.cols;
        int orig_h = frame.rows;
        int cx = orig_w / 2;
        int cy = orig_h / 2;

        // 640x640 Letterbox
        frame.copyTo(letterbox(cv::Rect(0, y_off, orig_w, orig_h)));

        // Tensör Bağlama ve Çıkarım
        ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, letterbox.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        ov::Tensor output_tensor = infer_request.get_output_tensor(0);
        const float* out_ptr = output_tensor.data<float>();
        auto shape = output_tensor.get_shape(); // [1, 5, 8400]

        int channels = shape[1]; // 4 coord + 1 class = 5
        int anchors  = shape[2]; // 8400

        float cur_thresh = target.active ? CONF_HOLD : CONF_ACQUIRE;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        // YOLOv8 Çıktı Çözümleme: [1, 5, 8400] bellekte ardışık kanallar halindedir
        const float* ptr_cx = out_ptr + 0 * anchors;
        const float* ptr_cy = out_ptr + 1 * anchors;
        const float* ptr_w  = out_ptr + 2 * anchors;
        const float* ptr_h  = out_ptr + 3 * anchors;
        const float* ptr_sc = out_ptr + 4 * anchors; // Dart güven skoru

        for (int i = 0; i < anchors; ++i) {
            float score = ptr_sc[i];
            if (score > cur_thresh) {
                float x_c = ptr_cx[i];
                float y_c = ptr_cy[i];
                float bw  = ptr_w[i];
                float bh  = ptr_h[i];

                int left = std::max(0, std::min(orig_w, (int)(x_c - bw / 2.0f)));
                int top  = std::max(0, std::min(orig_h, (int)((y_c - bh / 2.0f) - y_off)));
                int w_box = (int)bw;
                int h_box = (int)bh;

                if (left + w_box > orig_w) w_box = orig_w - left;
                if (top + h_box > orig_h) h_box = orig_h - top;

                if (w_box < 10 || h_box < 10) continue;

                // Dart geometri filtresi
                float aspect = (float)std::max(w_box, h_box) / (float)std::min(w_box, h_box);
                if (!target.active && (w_box * h_box) > 250 && aspect < 1.30f) {
                    continue;
                }

                boxes.push_back(cv::Rect(left, top, w_box, h_box));
                confidences.push_back(score);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, cur_thresh, 0.35f, indices);

        int best_idx = -1;
        float max_s = 0.0f;

        if (!indices.empty()) {
            if (target.active) {
                int cur_cx = target.box.x + target.box.width / 2;
                int cur_cy = target.box.y + target.box.height / 2;
                float min_dist = 140.0f;

                for (int idx : indices) {
                    int n_cx = boxes[idx].x + boxes[idx].width / 2;
                    int n_cy = boxes[idx].y + boxes[idx].height / 2;
                    float dist = std::hypot(n_cx - cur_cx, n_cy - cur_cy);

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = idx;
                    }
                }
            } else {
                for (int idx : indices) {
                    if (confidences[idx] > max_s) {
                        max_s = confidences[idx];
                        best_idx = idx;
                    }
                }
            }
        }

        // Kilit Takip Mantığı
        if (best_idx != -1) {
            cv::Rect b = boxes[best_idx];
            float c = confidences[best_idx];

            if (!target.active) {
                target.box = b;
                target.conf = c;
                target.active = true;
            } else {
                target.box.x = (int)(0.85f * b.x + 0.15f * target.box.x);
                target.box.y = (int)(0.85f * b.y + 0.15f * target.box.y);
                target.box.width = (int)(0.85f * b.width + 0.15f * target.box.width);
                target.box.height = (int)(0.85f * b.height + 0.15f * target.box.height);
                target.conf = c;
            }
            target.lost = 0;
        } else {
            if (target.active) {
                target.lost++;
                if (target.lost > MAX_LOST) {
                    target.active = false;
                }
            }
        }

        // Çizim
        if (target.active) {
            int tx = target.box.x + target.box.width / 2;
            int ty = target.box.y + target.box.height / 2;

            cv::Scalar col = (target.lost == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            std::string st = (target.lost == 0) ? "LOCKED " : "HOLD ";
            st += std::to_string((int)(target.conf * 100)) + "%";

            cv::rectangle(frame, target.box, col, 2);
            cv::circle(frame, cv::Point(tx, ty), 4, cv::Scalar(0, 0, 255), -1);
            cv::line(frame, cv::Point(cx, cy), cv::Point(tx, ty), cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, st, cv::Point(target.box.x, std::max(22, target.box.y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);

            int dx = tx - cx;
            int dy = ty - cy;
            cv::putText(frame, "DX: " + std::to_string(dx) + " | DY: " + std::to_string(dy),
                        cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
        } else {
            cv::putText(frame, "SEARCHING...", cv::Point(20, 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 165, 255), 2);
        }

        cv::drawMarker(frame, cv::Point(cx, cy), cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 20, 2);

        // Donanım Zamanlamalı FPS
        auto t_now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(t_now - t_prev).count();
        t_prev = t_now;
        if (dt > 0.0f) fps = 0.85f * fps + 0.15f * (1.0f / dt);

        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Nerf Interceptor Native OpenVINO", frame);
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
