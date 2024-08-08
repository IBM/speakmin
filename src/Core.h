#ifndef CORE_H
#define CORE_H

#include <vector>
#include <queue>
#include <string>
#include "Neuron.h"
#include "Spike.h"
#include "Event_unit.h"
#include "Config.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Core {
public:
    Core(const std::string& param_file, const std::string& weights_file);
    Core(const std::string& param_file, const std::string& weights_file, const std::vector<int>& tau_values);
    Core(const Config& config);
    Core(const Config& config, const std::vector<int>& tau_values);
    Core(const Core& other); // 복사 생성자
    Core& operator=(const Core& other); // 복사 대입 연산자

    bool run(uint32_t input_T_sim);
    void load_spike_train(const std::vector<uint32_t>& spike_times, const std::vector<uint16_t>& neuron_indices);
    void save_recorded_spikes(const std::string& filename);
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);

    void reset(); // 초기화 함수 추가

    bool enabling_train;
    uint8_t class_label;
    size_t train_index = 0;
    int N_training_times = 4;
    double lr;

private:
    std::vector<std::vector<double>> W_in, W_res, W_out, W_bias;
    std::vector<std::vector<bool>> W_fb;
    std::vector<Neuron> Neu_res, Neu_out, Neu_bias;
    std::vector<uint8_t> Neu_acc;
    // std::priority_queue<Spike_input> external_S_queue;
    std::priority_queue<Spike> external_S_queue;
    std::priority_queue<Spike> internal_S_queue;
    std::priority_queue<Spike> S_vec_trace;
    std::priority_queue<Spike> S_vec_trace_delay;
    std::priority_queue<Event_unit> Event_queue;
    std::vector<Spike> S_vec_now;
    std::vector<Spike> S_vec_trace_now;
    std::vector<Spike> S_vec_trace_delay_now;
    std::vector<Event_unit> Event_vec_now;
    std::priority_queue<Event_unit> Event_queue_delay;
    uint32_t T_sim; // 수정된 멤버 변수
    uint32_t t_delay; // 수정된 멤버 변수
    int N_out_times;

    std::vector<uint32_t> recorded_times; // 추가된 멤버 변수
    std::vector<uint16_t> recorded_neuron_indices; // 추가된 멤버 변수

    bool run_loop(uint32_t T_sim);
    void record_spike(uint32_t time, int neuron_index);
};

#endif // CORE_H
