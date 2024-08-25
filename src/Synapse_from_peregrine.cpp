// (C) Copyright IBM Corp. 2017
#include <iostream>
#include <algorithm>

#include "smsim.hpp"
#include "sm_core.hpp"

#ifdef DEBUG_WEIGHT
#define WT_DUMP(str, i, j)	\
	printf("weight_update index_a: %d index_b: %d %s %.19f %.19f\n", i, j, str, weight_matrix[i][j].Gp, weight_matrix[i][j].Gm);
#define WT_DUMP_BEFORE(i, j)	WT_DUMP("before", i, j)
#define WT_DUMP_AFTER(i, j)	WT_DUMP("after ", i, j)
#else
#define WT_DUMP_BEFORE(i, j)
#define WT_DUMP_AFTER(i, j)
#endif

#ifdef DEBUG_SHIFT_RESET
#define DEBUG_WEIGHT
#define WT_DUMP_SR_BEFORE(i, j) WT_DUMP("SR_before", i, j)
#define WT_DUMP_SR_AFTER(i, j)  WT_DUMP("SR_after ", i, j)
#else
#define WT_DUMP_SR_BEFORE(i, j)
#define WT_DUMP_SR_AFTER(i, j)
#endif

inline void sm_core::weight_set_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	double &weight = weight_matrix[v_idx][h_idx].Gp;
	double wt_delta_g = wt_delta_g_set[v_idx][h_idx].Gp;
	double max_w = max_weight[v_idx][h_idx].Gp;
	double min_w = min_weight[v_idx][h_idx].Gp;

	if(rng_wt_set) {
		double &ideal = weight_matrix_ideal[v_idx][h_idx].Gp;
		ideal += wt_delta_g;
		if(weight > max_w) ideal = max_w;
		weight = ideal + rng_wt_set->get_val();
	} else {
		weight += wt_delta_g;
	}

	if(weight > max_w) {
		weight = max_w;
	} else if(weight < min_w) {
		weight = min_w;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_set_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	double &weight = weight_matrix[v_idx][h_idx].Gm;
	double wt_delta_g = wt_delta_g_set[v_idx][h_idx].Gm;
	double max_w = max_weight[v_idx][h_idx].Gm;
	double min_w = min_weight[v_idx][h_idx].Gm;

	if(rng_wt_set) {
		double &ideal = weight_matrix_ideal[v_idx][h_idx].Gm;
		ideal += wt_delta_g;
		if(weight > max_w) ideal = max_w;
		weight = ideal + rng_wt_set->get_val();
	} else {
		weight += wt_delta_g;
	}

	if(weight > max_w) {
		weight = max_w;
	} else if(weight < min_w) {
		weight = min_w;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_set_pcm_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	int step = weight_step_matrix[v_idx][h_idx].Gp + 1;
	if(step >= param.pcm_model_steps) step = param.pcm_model_steps - 1;
	double weight = pcm_set_model[step];
	if(rng_wt_set) {
		weight_matrix[v_idx][h_idx].Gp = weight + rng_wt_set->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gp = weight;
	}
	weight_step_matrix[v_idx][h_idx].Gp = step;

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_set_pcm_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	int step = weight_step_matrix[v_idx][h_idx].Gm + 1;
	if(step >= param.pcm_model_steps) step = param.pcm_model_steps - 1;
	double weight = pcm_set_model[step];
	if(rng_wt_set) {
		weight_matrix[v_idx][h_idx].Gm = weight + rng_wt_set->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gm = weight;
	}
	weight_step_matrix[v_idx][h_idx].Gm = step;

	WT_DUMP_AFTER(v_idx, h_idx);
}

// pcm equation model -- from here --
inline void sm_core::weight_set_pcm_eq_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	double G = weight_matrix[v_idx][h_idx].Gp;
	pcm_eq_model_pmem[v_idx][h_idx].Gp = pcm_eq_model_pmem[v_idx][h_idx].Gp * param.pcm_eq_model_alpha_exp;
	double mu_dG = param.pcm_eq_model_m1 * G + param.pcm_eq_model_c1 + param.pcm_eq_model_a1 * pcm_eq_model_pmem[v_idx][h_idx].Gp;
	double sigma_dG = param.pcm_eq_model_m2 * G + param.pcm_eq_model_c2 + param.pcm_eq_model_a2 * pcm_eq_model_pmem[v_idx][h_idx].Gp;
	double newG;
	while(true){
		double dG = mu_dG + sigma_dG * rng_pcm_eq_model_set->get_val();
		newG = G + dG;
		if(newG > 0){
			break;
		}
	}

	if( newG > max_weight[v_idx][h_idx].Gp){
		weight_matrix[v_idx][h_idx].Gp = max_weight[v_idx][h_idx].Gp;
	} else {
		weight_matrix[v_idx][h_idx].Gp = newG;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_set_pcm_eq_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	double G = weight_matrix[v_idx][h_idx].Gm;
	pcm_eq_model_pmem[v_idx][h_idx].Gm = pcm_eq_model_pmem[v_idx][h_idx].Gm * param.pcm_eq_model_alpha_exp;
	double mu_dG = param.pcm_eq_model_m1 * G + param.pcm_eq_model_c1 + param.pcm_eq_model_a1 * pcm_eq_model_pmem[v_idx][h_idx].Gm;
	double sigma_dG = param.pcm_eq_model_m2 * G + param.pcm_eq_model_c2 + param.pcm_eq_model_a2 * pcm_eq_model_pmem[v_idx][h_idx].Gm;
	double newG;
	while(true){
		double dG = mu_dG + sigma_dG * rng_pcm_eq_model_set->get_val();
		newG = G + dG;
		if(newG > 0){
			break;
		}
	}

	if( newG > max_weight[v_idx][h_idx].Gm){
		weight_matrix[v_idx][h_idx].Gm = max_weight[v_idx][h_idx].Gm;
	} else {
		weight_matrix[v_idx][h_idx].Gm = newG;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}
// pcm equation model -- to here --

inline void sm_core::weight_reset_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	double &weight = weight_matrix[v_idx][h_idx].Gp;
	double wt_delta_g = wt_delta_g_reset[v_idx][h_idx].Gp;
	double max_w = max_weight[v_idx][h_idx].Gp;
	double min_w = min_weight[v_idx][h_idx].Gp;

	if(rng_wt_reset) {
		double &ideal = weight_matrix_ideal[v_idx][h_idx].Gp;
		ideal += wt_delta_g;
		if(weight < min_w) ideal = min_w;
		weight = ideal + rng_wt_reset->get_val();
	} else {
		weight += wt_delta_g;
	}

	if(weight > max_w) {
		weight = max_w;
	} else if(weight < min_w) {
		weight = min_w;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_reset_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	double &weight = weight_matrix[v_idx][h_idx].Gm;
	double wt_delta_g = wt_delta_g_reset[v_idx][h_idx].Gm;
	double max_w = max_weight[v_idx][h_idx].Gm;
	double min_w = min_weight[v_idx][h_idx].Gm;

	if(rng_wt_reset) {
		double &ideal = weight_matrix_ideal[v_idx][h_idx].Gm;
		ideal += wt_delta_g;
		if(weight < min_w) ideal = min_w;
		weight = ideal + rng_wt_reset->get_val();
	} else {
		weight += wt_delta_g;
	}

	if(weight > max_w) {
		weight = max_w;
	} else if(weight < min_w) {
		weight = min_w;
	}

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_reset_pcm_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(param.max_weight > -param.wt_delta_g_reset) {
		cerr << "Absolute value of wt_delta_g_reset is smaller than max_weight. Not supported." << endl;
		exit(0);
	}

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	int step = param.pcm_model_steps_min;
	double weight = pcm_set_model[step];
	if(rng_wt_reset) {
		weight_matrix[v_idx][h_idx].Gp = weight + rng_wt_reset->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gp = weight;
	}
	weight_step_matrix[v_idx][h_idx].Gp = step;

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_reset_pcm_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(param.max_weight > -param.wt_delta_g_reset) {
		cerr << "Absolute value of wt_delta_g_reset is smaller than max_weight. Not supported." << endl;
		exit(0);
	}

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	int step = param.pcm_model_steps_min;
	double weight = pcm_set_model[step];
	if(rng_wt_reset) {
		weight_matrix[v_idx][h_idx].Gm = weight + rng_wt_reset->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gm = weight;
	}
	weight_step_matrix[v_idx][h_idx].Gm = step;

	WT_DUMP_AFTER(v_idx, h_idx);
}

// pcm equation model -- from here --
inline void sm_core::weight_reset_pcm_eq_gp(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	if(rng_pcm_eq_model_reset){
		weight_matrix[v_idx][h_idx].Gp = param.pcm_eq_model_g_reset_min + rng_pcm_eq_model_reset->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gp = param.pcm_eq_model_g_reset_min;
	}
	pcm_eq_model_pmem[v_idx][h_idx].Gp = interpolate(pcm_eq_model_G_vs_Pmem__G, pcm_eq_model_G_vs_Pmem__Pmem,
																									 weight_matrix[v_idx][h_idx].Gp, false); // No extrapolate

	WT_DUMP_AFTER(v_idx, h_idx);
}

inline void sm_core::weight_reset_pcm_eq_gm(int v_idx, int h_idx) {
	WT_DUMP_BEFORE(v_idx, h_idx);

	if(rng_wt_reset_rate) {
		if(param.wt_reset_rate < rng_wt_reset_rate->get_val()) {
			return;
		}
	}

	if(rng_pcm_eq_model_reset){
		weight_matrix[v_idx][h_idx].Gm = param.pcm_eq_model_g_reset_min + rng_pcm_eq_model_reset->get_val();
	} else {
		weight_matrix[v_idx][h_idx].Gm = param.pcm_eq_model_g_reset_min;
	}
	pcm_eq_model_pmem[v_idx][h_idx].Gm = interpolate(pcm_eq_model_G_vs_Pmem__G, pcm_eq_model_G_vs_Pmem__Pmem,
																									 weight_matrix[v_idx][h_idx].Gm, false); // No extrapolate

	WT_DUMP_AFTER(v_idx, h_idx);
}
// pcm equation model -- to here --

void sm_core::weight_update_bipolar(int phase, int side, int spk_idx) {
	if(side == side_v) {
		if(last_spk[side_v][spk_idx] < 0) { return; }
		if(phase == sm_data_phase) {
			for(int i = 0; i < num_neurons[side_h]; i++) {
				if(last_spk[side_h][i] < 0) { continue; }
				double td = last_spk[side_v][spk_idx] - last_spk[side_h][i];
				if((fabs(td) >= param.stdp_window) || (td == 0.0)) { continue; }

				weight_set_gp(spk_idx, i);
			}
		} else {
			for(int i = 0; i < num_neurons[side_h]; i++) {
				if(last_spk[side_h][i] < 0) { continue; }
				double td = last_spk[side_v][spk_idx] - last_spk[side_h][i];
				if((fabs(td) >= param.stdp_window) || (td == 0.0)) { continue; }

				weight_reset_gp(spk_idx, i);
			}
		}
	} else { // side_h
		if(last_spk[side_h][spk_idx] < 0) { return; }
		if(phase == sm_data_phase) {
			for(int i = 0; i < num_neurons[side_v]; i++) {
				if(last_spk[side_v][i] < 0) { continue; }
				double td = last_spk[side_h][spk_idx] - last_spk[side_v][i];
				if((fabs(td) >= param.stdp_window) || (td == 0.0)) { continue; }

				weight_set_gp(i, spk_idx);
			}
		} else {
			for(int i = 0; i < num_neurons[side_v]; i++) {
				if(last_spk[side_v][i] < 0) { continue; }
				double td = last_spk[side_h][spk_idx] - last_spk[side_v][i];
				if((fabs(td) >= param.stdp_window) || (td == 0.0)) { continue; }

				weight_reset_gp(i, spk_idx);
			}
		}
	}
}

void sm_core::weight_update_gpgm(int phase, int side, int spk_idx, double wup_time) {
	if(side == side_v) return;
	if(phase == sm_data_phase) {
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_gp(i, spk_idx);
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.treset_width)) {
				weight_reset_gm(i, spk_idx);
			}
		}
	} else { // model phase
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl + param.tset_width - param.treset_width) &&
				(wup_time < time_stdp_wl + param.tset_width)) {
				weight_reset_gp(i, spk_idx);
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_gm(i, spk_idx);
			}
		}
	}
}

// Refresh phase method
void sm_core::weight_update_pcm(int phase, int side, int spk_idx, double wup_time) {
	if(side == side_v) return;
	if(phase == sm_data_phase) {
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_pcm_gp(i, spk_idx);
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.treset_width)) {
				weight_reset_pcm_gm(i, spk_idx);
			}
		}
	} else { // model phase
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl + param.tset_width - param.treset_width) &&
				(wup_time < time_stdp_wl + param.tset_width)) {
				weight_reset_pcm_gp(i, spk_idx);
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_pcm_gm(i, spk_idx);
			}
		}
	}
}

void sm_core::weight_update_pcm_eq(int phase, int side, int spk_idx, double wup_time) {
	if(side == side_v) return;
	if(phase == sm_data_phase) {
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_pcm_eq_gp(i, spk_idx);
				pcm_eq_model_tp[i][spk_idx].Gp = wup_time;
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.treset_width)) {
				weight_reset_pcm_eq_gm(i, spk_idx);
				pcm_eq_model_tp[i][spk_idx].Gm = wup_time;
			}
		}
	} else { // model phase
		for(int i = 0; i < num_neurons[side_v]; i++) {
			if(last_spk[side_v][i] < 0) { continue; }

			double time_stdp_wl = last_spk[side_v][i];
			if((wup_time >= time_stdp_wl + param.tset_width - param.treset_width) &&
				(wup_time < time_stdp_wl + param.tset_width)) {
				weight_reset_pcm_eq_gp(i, spk_idx);
				pcm_eq_model_tp[i][spk_idx].Gp = wup_time;
			}
			if((wup_time >= time_stdp_wl) && (wup_time < time_stdp_wl + param.tset_width)) {
				weight_set_pcm_eq_gm(i, spk_idx);
				pcm_eq_model_tp[i][spk_idx].Gm = wup_time;
			}
		}
	}
}

void sm_core::weight_update(double spk_now_time) {
	if(queue_wup_ext.empty() && queue_wup_spk.empty()) {
		return;
	}

	// Update weights which event times are passed
	sm_spk *wup;
	sm_spk *wup_ext;
	sm_spk *wup_spk;
	priority_queue<pair<double, sm_spk*>, vector<pair<double, sm_spk*>>, spk_cmp> *queue_wup;
	while(1) {
		if(queue_wup_ext.empty()) {
			if(queue_wup_spk.empty()) {
				break;
			} else {
				wup = queue_wup_spk.top().second;
				queue_wup = &queue_wup_spk;
			}
		} else {
			if(queue_wup_spk.empty()) {
				wup = queue_wup_ext.top().second;
				queue_wup = &queue_wup_ext;
			} else {
				wup_ext = queue_wup_ext.top().second;
				wup_spk = queue_wup_spk.top().second;
				if(wup_ext->time < wup_spk->time) {
					wup = wup_ext;
					queue_wup = &queue_wup_ext;
				} else {
					wup = wup_spk;
					queue_wup = &queue_wup_spk;
				}
			}
		}
		if(wup->time > spk_now_time) {
			// weight update done
			break;
		}

		// Update last_spk before updating weights
		for(auto it = wup->spk.begin(); it != wup->spk.end(); it++) {
			last_spk[it->first][it->second] = wup->time;
		}

		// Update weights
		int sm_phase = phase->query_phase(wup->time);
#ifdef DEBUG_WEIGHT
		cout << "weight update time " << wup->time << " phase " << sm_phase << endl;
#endif

		if(sm_phase != sm_transition_phase) {
			for(auto it = wup->spk.begin(); it != wup->spk.end(); it++) {
				int spk_idx = it->second;

				if(param.enable_gpgm) {
					if(param.enable_pcm_model) {
						weight_update_pcm(sm_phase, it->first, it->second, wup->time);
					} else if(param.enable_pcm_eq_model){
						weight_update_pcm_eq(sm_phase, it->first, it->second, wup->time);
					} else {
						weight_update_gpgm(sm_phase, it->first, it->second, wup->time);
					}
				} else {
					weight_update_bipolar(sm_phase, it->first, it->second);
				}
			}
		}
		delete queue_wup->top().second;
		queue_wup->pop();
	}
}

inline double sm_core::weight_discretize(double weight){
	return wt_rw_per_step * ((int)((weight + wt_rw_per_step_half) / wt_rw_per_step));
}

void sm_core::shift_reset(int v_idx, int h_idx, double G, double tnow){
	int step = 0;
	if(param.enable_pcm_model) {
		int refresh_step_idx = (int)(fabs(G) / wt_rw_per_step);
		step = param.wt_rw_refresh_steps[refresh_step_idx];
		double val = pcm_set_model[step];
		if(G > 0) {
			weight_matrix[v_idx][h_idx].Gm = param.min_weight;
			if(rng_wt_set) {
				weight_matrix[v_idx][h_idx].Gp = val + rng_wt_set->get_val();
			} else {
				weight_matrix[v_idx][h_idx].Gp = val;
			}
			weight_step_matrix[v_idx][h_idx].Gm = param.pcm_model_steps_min;
			weight_step_matrix[v_idx][h_idx].Gp = step;
		} else if(G < 0) {
			if(rng_wt_set) {
				weight_matrix[v_idx][h_idx].Gm = val + rng_wt_set->get_val();
			} else {
				weight_matrix[v_idx][h_idx].Gm = val;
			}
			weight_matrix[v_idx][h_idx].Gp = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gm = step;
			weight_step_matrix[v_idx][h_idx].Gp = param.pcm_model_steps_min;
		} else {
			weight_matrix[v_idx][h_idx].Gm = param.min_weight;
			weight_matrix[v_idx][h_idx].Gp = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gm = param.pcm_model_steps_min;
			weight_step_matrix[v_idx][h_idx].Gp = param.pcm_model_steps_min;
		}
	} else if(param.enable_pcm_eq_model) {
		double Gp, Gm;
		pcm_eq_model_tp[v_idx][h_idx].Gp = tnow;
		pcm_eq_model_tp[v_idx][h_idx].Gm = tnow;
		if(G > 0) {
			// Gm -------------------------
			if( param.pcm_eq_model_halfway_shift_reset ) { // halfway shift-reset
				if(G > pcm_eq_model_half_weight){
					Gm = param.pcm_eq_model_g_reset_min;
					weight_reset_pcm_eq_gm(v_idx, h_idx);
				} else {
					Gm = pcm_eq_model_half_weight;
					weight_matrix[v_idx][h_idx].Gm = Gm;
					pcm_eq_model_pmem[v_idx][h_idx].Gm = pcm_eq_model_pmem_at_half_weight;
				}
			} else { // normal shift-reset
				Gm = param.pcm_eq_model_g_reset_min;
				weight_reset_pcm_eq_gm(v_idx, h_idx);
			}
			// Gp -------------------------
			Gp = Gm + G;
			pcm_eq_model_pmem[v_idx][h_idx].Gp = interpolate(pcm_eq_model_G_vs_Pmem__G, pcm_eq_model_G_vs_Pmem__Pmem,
																											 Gp, false); // No extrapolate
			double sigma_dG = param.pcm_eq_model_m2 * Gp + param.pcm_eq_model_c2
				+ param.pcm_eq_model_a2 * pcm_eq_model_pmem[v_idx][h_idx].Gp;

			if( sigma_dG != 0){
				while(true){
					Gp = Gp + sigma_dG * rng_pcm_eq_model_set->get_val();
					if(param.min_weight < Gp && Gp < param.max_weight){
						break;
					}
				}
			}

			if( Gp > max_weight[v_idx][h_idx].Gp){
				weight_matrix[v_idx][h_idx].Gp = max_weight[v_idx][h_idx].Gp;
			} else {
				weight_matrix[v_idx][h_idx].Gp = Gp;
			}
		} else if(G < 0) {
			G = -G; // change to absolute value
			// Gp -------------------------
			if( param.pcm_eq_model_halfway_shift_reset ) { // halfway shift-reset
				if(G > pcm_eq_model_half_weight){
					Gp = param.pcm_eq_model_g_reset_min;
					weight_reset_pcm_eq_gp(v_idx, h_idx);
				} else {
					Gp = pcm_eq_model_half_weight;
					weight_matrix[v_idx][h_idx].Gp = Gp;
					pcm_eq_model_pmem[v_idx][h_idx].Gp = pcm_eq_model_pmem_at_half_weight;
				}
			} else { // normal shift-reset
				Gp = param.pcm_eq_model_g_reset_min;
				weight_reset_pcm_eq_gp(v_idx, h_idx);
			}
			// Gm -------------------------
			Gm = Gp + G;
			pcm_eq_model_pmem[v_idx][h_idx].Gm = interpolate(pcm_eq_model_G_vs_Pmem__G, pcm_eq_model_G_vs_Pmem__Pmem,
																											 Gm, false); // No extrapolate
			double sigma_dG = param.pcm_eq_model_m2 * Gm + param.pcm_eq_model_c2
				+ param.pcm_eq_model_a2 * pcm_eq_model_pmem[v_idx][h_idx].Gm;

			if( sigma_dG != 0){
				while(true){
					Gm = Gm + sigma_dG * rng_pcm_eq_model_set->get_val();
					if(param.min_weight < Gm && Gm < param.max_weight){
						break;
					}
				}
			}

			if( Gm > max_weight[v_idx][h_idx].Gm){
				weight_matrix[v_idx][h_idx].Gm = max_weight[v_idx][h_idx].Gm;
			} else {
				weight_matrix[v_idx][h_idx].Gm = Gm;
			}
		} else {
			if( param.pcm_eq_model_halfway_shift_reset ) { // halfway shift-reset
				weight_matrix[v_idx][h_idx].Gp = pcm_eq_model_half_weight;
				weight_matrix[v_idx][h_idx].Gm = pcm_eq_model_half_weight;
				pcm_eq_model_pmem[v_idx][h_idx].Gp = pcm_eq_model_pmem_at_half_weight;
				pcm_eq_model_pmem[v_idx][h_idx].Gm = pcm_eq_model_pmem_at_half_weight;
			} else {
				weight_reset_pcm_eq_gp(v_idx, h_idx);
				weight_reset_pcm_eq_gm(v_idx, h_idx);
			}
		}
	} else {
		if(G > 0) {
			weight_matrix[v_idx][h_idx].Gm = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gm = 0;
			step = (int)(G / param.wt_delta_g_set);
			double weight = wt_delta_g_set[v_idx][h_idx].Gp * step;
			if(weight > param.max_weight) weight = param.max_weight;
			weight_matrix[v_idx][h_idx].Gp = weight;
			weight_step_matrix[v_idx][h_idx].Gp = step;
		} else if(G < 0) {
			step = (int)(-G / param.wt_delta_g_set);
			double weight = wt_delta_g_set[v_idx][h_idx].Gm * step;
			if(weight > param.max_weight) weight = param.max_weight;
			weight_matrix[v_idx][h_idx].Gm = weight;
			weight_step_matrix[v_idx][h_idx].Gm = step;
			weight_matrix[v_idx][h_idx].Gp = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gp = 0;
		} else {
			weight_matrix[v_idx][h_idx].Gm = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gm = 0;
			weight_matrix[v_idx][h_idx].Gp = param.min_weight;
			weight_step_matrix[v_idx][h_idx].Gp = 0;
		}
	}

	if(rng_wt_set || rng_wt_reset) {
		weight_matrix_ideal[v_idx][h_idx].Gm = weight_matrix[v_idx][h_idx].Gm;
		weight_matrix_ideal[v_idx][h_idx].Gp = weight_matrix[v_idx][h_idx].Gp;
	}

	if(param.dump_shift_reset) {
		cout << (int)((fabs(G) + wt_rw_per_step_half) / wt_rw_per_step) << endl;
	}
	if(param.count_shift_reset) {
		pcm_set_count += step;
		pcm_reset_count += 2; // reset count - Gp and Gm
	}
}

void sm_core::occasional_reset(double tnow, sm_rng_ureal01 *rng) {
	if(!param.enable_gpgm) {
		return;
	}

	int num_of_reset_neuron = param.num_of_reset_synapse;
	if(param.reset_all_neurons) num_of_reset_neuron = num_neurons[side_v];
	for(int i = 0; i < num_of_reset_neuron; i++) { // number of axon(visible) selection
		int v_idx;
		if(param.reset_all_neurons) {
			v_idx = i;
		} else {
			v_idx = int(rng->get_val() * num_neurons[side_v]);
		}

		for(int h_idx = 0; h_idx < num_neurons[side_h]; h_idx++) { // all neurons(hidden)
			WT_DUMP_SR_BEFORE(v_idx, h_idx);
			double Gp = weight_matrix[v_idx][h_idx].Gp;
			double Gm = weight_matrix[v_idx][h_idx].Gm;

			if((wt_rw_reset_target_threshold <= Gp or wt_rw_reset_target_threshold <= Gm) ||
				param.reset_all_synapses) { // target selection
				if(param.wt_rw_strong_reset_enable) { // Gp & Gm both strong reset
					weight_matrix[v_idx][h_idx].Gm = param.min_weight;
					weight_matrix[v_idx][h_idx].Gp = param.min_weight;
				} else { // shift reset
					if(param.wt_rw_num_of_steps != 0){ // discretize
						Gp = weight_discretize(Gp);
						Gm = weight_discretize(Gm);
					}
					double G = Gp - Gm;
					shift_reset(v_idx, h_idx, G, tnow);
				}
			}
			WT_DUMP_SR_AFTER(v_idx, h_idx);
		}
	}
}