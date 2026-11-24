#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hermite_rules.hpp"

namespace {
constexpr double inverse_sqrt_two = 0.70710678118654752440084436210485;
constexpr double guard = 1e-10;
constexpr int solver_count = 4;

struct State {
    double spot, average, fixings, rate, t1, t2, vol1, vol2, jump_weight, jump_multiplier, jump_vol;
};
struct Component {
    double weight, log_location, sigma, forward;
};
struct Context {
    double payoff, disc1, disc2, inv_final_n, inv_middle_n, support, time_rate2, sigma2, at_zero;
    int components;
    std::array<Component, 2> c;
};
struct Decision {
    bool continue_now, certified;
    int calls;
    double lower = -1, upper = -1;
};

double cdf(double x) {
    return 0.5 * std::erfc(-x * inverse_sqrt_two);
}

Component distribution(const State& s, double multiplier, double vol, double weight) {
    const double sigma = vol * std::sqrt(s.t1);
    const double loc = std::log(s.spot * multiplier) + s.rate * s.t1 - 0.5 * sigma * sigma;
    return {weight, loc, sigma, s.spot * multiplier * std::exp(s.rate * s.t1)};
}

Context setup(const State& s) {
    const double accrued = s.average * s.fixings;
    Context x{};
    x.payoff = 1 - s.average;
    x.time_rate2 = s.rate * s.t2;
    x.disc1 = std::exp(-s.rate * s.t1);
    x.disc2 = std::exp(-x.time_rate2);
    x.inv_middle_n = 1 / (s.fixings + 1);
    x.inv_final_n = 1 / (s.fixings + 2);
    x.support = s.fixings + 2 - accrued;
    x.sigma2 = s.vol2 * std::sqrt(s.t2);
    x.at_zero = std::max((x.support - 1) * x.inv_middle_n, x.disc2 * x.support * x.inv_final_n);
    x.components = s.jump_weight > 0 ? 2 : 1;
    const double regular_multiple = (1 - s.jump_weight * s.jump_multiplier) / (1 - s.jump_weight);
    x.c[0] = distribution(s, regular_multiple, s.vol1, 1 - s.jump_weight);
    if (x.components == 2) {
        x.c[1] = distribution(s, s.jump_multiplier, s.jump_vol, s.jump_weight);
    }
    return x;
}

double value(const Context& x, double spot) {
    const double strike = x.support - spot;
    if (strike <= 0) {
        return 0;
    }
    if (spot <= 0) {
        return x.at_zero;
    }
    const double d1 = (std::log(spot / strike) + x.time_rate2) / x.sigma2 + 0.5 * x.sigma2;
    const double wait = (x.disc2 * strike * cdf(x.sigma2 - d1) - spot * cdf(-d1)) * x.inv_final_n;
    const double exercise = std::max(0.0, (strike - 1) * x.inv_middle_n);
    return std::max(exercise, wait);
}

template <std::size_t N>
double gh(const Context& x, const std::array<double, N>& nodes, const std::array<double, N>& weights) {
    double result = 0;
    for (int j = 0; j < x.components; ++j) {
        const Component& c = x.c[j];
        double mid = 0;
        for (std::size_t k = 0; k < N; ++k) {
            const double spot = std::exp(c.log_location + c.sigma * nodes[k]);
            mid += weights[k] * value(x, spot);
        }
        result += c.weight * mid;
    }
    return x.disc1 * result;
}

struct Edge {
    double spot, probability, moment, val;
};
struct Leaf {
    Edge a, b;
    double mean, at_mean, lower, upper;
    int comp;
};

Edge edge(const Component& c, double spot, double val) {
    const double z = (std::log(spot) - c.log_location) / c.sigma;
    return {spot, cdf(z), c.forward * cdf(z - c.sigma), val};
}

Leaf leaf(const Context& x, int comp, Edge a, Edge b) {
    const double mass = b.probability - a.probability;
    const double width = b.spot - a.spot;
    const double mean = mass > 0 ? std::clamp((b.moment - a.moment) / mass, a.spot, b.spot) : a.spot;
    const double mid = value(x, mean);
    const double alpha = width > 0 ? (mean - a.spot) / width : 0;
    const double chord = a.val + alpha * (b.val - a.val);
    const double scale = x.disc1 * x.c[comp].weight * mass;
    return {a, b, mean, mid, scale * mid, scale * std::max(mid, chord), comp};
}

Leaf first_leaf(const Context& x, int comp) {
    return leaf(x, comp, {0, 0, 0, x.at_zero}, edge(x.c[comp], x.support, 0));
}

bool proof(double payoff, double lower, double upper) {
    return lower > payoff + guard || upper < payoff - guard;
}

Decision zero_payoff_decision() {
    return {true, true, 0};
}

Decision bracket_decision(const Context& x, double lower, double upper, int calls) {
    const bool certified = proof(x.payoff, lower, upper);
    return {0.5 * (lower + upper) > x.payoff, certified, calls, lower, upper};
}

Decision refine(const Context& x, int cap) {
    // The old conditional mean is inside the leaf and already evaluated.
    std::array<Leaf, 72> leaves;
    int used = x.components;
    int num_leaves = x.components;
    double lower = 0, upper = 0;
    for (int j = 0; j < x.components; ++j) {
        leaves[j] = first_leaf(x, j);
        lower += leaves[j].lower;
        upper += leaves[j].upper;
    }
    while (!proof(x.payoff, lower, upper) && used + 2 <= cap) {
        const auto found = std::max_element(leaves.begin(), leaves.begin() + num_leaves,
            [](const Leaf& a, const Leaf& b) { return a.upper - a.lower < b.upper - b.lower; });
        const Leaf old = *found;
        if (old.mean <= old.a.spot || old.mean >= old.b.spot) {
            break;
        }
        const Edge split = edge(x.c[old.comp], old.mean, old.at_mean);
        *found = leaf(x, old.comp, old.a, split);
        const Leaf right = leaf(x, old.comp, split, old.b);
        leaves[num_leaves++] = right;
        lower += found->lower + right.lower - old.lower;
        upper += found->upper + right.upper - old.upper;
        used += 2;
    }
    return bracket_decision(x, lower, upper, used);
}

int components(const State& s) {
    return s.average < 1 ? 1 + (s.jump_weight > 0) : 0;
}

void measured_candidate(const std::vector<State>& states, std::vector<Decision>& output) {
    int reserve = 0;
    for (const State& s : states) {
        reserve += components(s);
    }
    int remaining = 19 * reserve;
    for (std::size_t i = 0; i < states.size(); ++i) {
        const State& s = states[i];
        const int c = components(s);
        reserve -= c;
        if (c == 0) {
            output[i] = zero_payoff_decision();
        } else {
            output[i] = refine(setup(s), std::min(63 * c, remaining - reserve));
            remaining -= output[i].calls;
        }
    }
}

void measured_fixed(const std::vector<State>& states, std::vector<Decision>& output) {
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (components(states[i]) == 0) {
            output[i] = zero_payoff_decision();
        } else {
            const Context x = setup(states[i]);
            output[i] = {gh(x, nodes19, weights19) > x.payoff, false, 19 * x.components};
        }
    }
}

void measured_staged(const std::vector<State>& states, std::vector<Decision>& output) {
    struct Priority { std::size_t i; double margin; };
    std::vector<Priority> ranked;
    ranked.reserve(states.size());
    std::vector<Context> cache(states.size());
    int allowance = 0;
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (components(states[i]) == 0) {
            output[i] = zero_payoff_decision();
        } else {
            Context& x = cache[i] = setup(states[i]);
            const double margin = gh(x, nodes7, weights7) - x.payoff;
            output[i] = {margin > 0, false, 7 * x.components};
            ranked.push_back({i, std::abs(margin)});
            allowance += 12 * x.components;
        }
    }
    if (ranked.empty()) {
        return;
    }
    // Component count is constant within each fixture batch.
    const auto pivot = ranked.begin() + std::min<std::size_t>(
        ranked.size(), allowance / (32 * cache[ranked.front().i].components));
    auto ranking = [](const Priority& a, const Priority& b) {
        return a.margin == b.margin ? a.i < b.i : a.margin < b.margin;
    };
    if (pivot != ranked.end()) {
        std::nth_element(ranked.begin(), pivot, ranked.end(), ranking);
    }
    for (auto p = ranked.begin(); p != pivot; ++p) {
        const std::size_t i = p->i;
        const Context& x = cache[i];
        const int cost = 32 * x.components;
        if (cost > allowance) {
            continue;
        }
        allowance -= cost;
        output[i] = {gh(x, nodes32, weights32) > x.payoff, false, 39 * x.components};
    }
}

void measured_screened(const std::vector<State>& states, std::vector<Decision>& output) {
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (components(states[i]) == 0) {
            output[i] = zero_payoff_decision();
            continue;
        }
        const Context x = setup(states[i]);
        const Decision screen = refine(x, x.components);
        output[i] = screen;
        if (!screen.certified) {
            output[i].continue_now = gh(x, nodes18, weights18) > x.payoff;
            output[i].calls += 18 * x.components;
        }
    }
}

using Solver = void (*)(const std::vector<State>&, std::vector<Decision>&);
constexpr std::array<Solver, solver_count> methods = {
    measured_candidate, measured_fixed, measured_staged, measured_screened,
};

struct Batch {
    std::vector<State> states;
    std::array<std::vector<Decision>, solver_count> output;
    std::array<std::vector<double>, solver_count> ms;
};

std::vector<Batch> read_batches(const std::string& path) {
    std::ifstream input(path);
    if (!input.good()) {
        throw std::runtime_error("Fixture not readable");
    }
    std::size_t n_batches, n_states;
    input >> n_batches >> n_states;
    if (n_batches == 0 || n_batches > 64 || n_states == 0 || n_states > 4096) {
        throw std::runtime_error("Invalid fixture size");
    }
    std::vector<Batch> batches(n_batches);
    for (auto& b : batches) {
        b.states.resize(n_states);
        for (auto& o : b.output) {
            o.resize(n_states);
        }
        for (State& s : b.states) {
            input >> s.spot >> s.average >> s.fixings >> s.rate >> s.t1 >> s.t2 >> s.vol1 >> s.vol2
                  >> s.jump_weight >> s.jump_multiplier >> s.jump_vol;
            if (input.fail()) {
                throw std::runtime_error("Invalid fixture row");
            }
        }
    }
    return batches;
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: solver states.txt decisions.tsv timings.tsv repeats\n";
        return 1;
    }
    auto batches = read_batches(argv[1]);
    const int repeats = std::stoi(argv[4]);
    if (repeats < 1 || repeats > 120) {
        return 2;
    }
    std::ofstream decisions(argv[2]), timings(argv[3]);
    if (!decisions || !timings) {
        return 3;
    }
    std::mt19937 random(81367023);
    std::vector<std::size_t> batches_order(batches.size());
    std::iota(batches_order.begin(), batches_order.end(), 0);
    std::array<int, solver_count> methods_order;
    std::iota(methods_order.begin(), methods_order.end(), 0);
    for (int k = -2; k < repeats; ++k) {
        std::shuffle(batches_order.begin(), batches_order.end(), random);
        for (const std::size_t b : batches_order) {
            std::shuffle(methods_order.begin(), methods_order.end(), random);
            for (const int m : methods_order) {
                auto& batch = batches[b];
                const auto start = std::chrono::steady_clock::now();
                methods[m](batch.states, batch.output[m]);
                const auto end = std::chrono::steady_clock::now();
                if (k >= 0) {
                    batch.ms[m].push_back(std::chrono::duration<double, std::milli>(end - start).count());
                }
            }
        }
    }
    decisions << std::setprecision(17);
    timings << std::setprecision(17);
    for (std::size_t b = 0; b < batches.size(); ++b) {
        for (int m = 0; m < solver_count; ++m) {
            for (const Decision& d : batches[b].output[m]) {
                decisions << d.continue_now << ' ' << d.certified << ' ' << d.calls << ' '
                          << d.lower << ' ' << d.upper << '\n';
            }
            for (int k = 0; k < repeats; ++k) {
                timings << b << ' ' << m << ' ' << k << ' ' << batches[b].ms[m][k] << '\n';
            }
        }
    }
}
