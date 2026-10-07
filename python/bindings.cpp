#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/velocity.hpp"
#include "cfd/reductions.hpp"
#include "cfd/gradients.hpp"
#include "cfd/divergence.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"
#include "cfd/experiments.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_parallelcfd, m) {
    m.doc() = "ParallelCFD: OpenMP-Accelerated CFD Field Analysis and Post-Processing Toolkit";

    // Common types
    py::class_<cfd::Grid3D>(m, "Grid3D")
        .def(py::init<size_t, size_t, size_t, double, double, double, double, double, double>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
             py::arg("x0") = 0.0, py::arg("y0") = 0.0, py::arg("z0") = 0.0)
        .def_readwrite("nx", &cfd::Grid3D::nx)
        .def_readwrite("ny", &cfd::Grid3D::ny)
        .def_readwrite("nz", &cfd::Grid3D::nz)
        .def_readwrite("dx", &cfd::Grid3D::dx)
        .def_readwrite("dy", &cfd::Grid3D::dy)
        .def_readwrite("dz", &cfd::Grid3D::dz)
        .def_readwrite("x0", &cfd::Grid3D::x0)
        .def_readwrite("y0", &cfd::Grid3D::y0)
        .def_readwrite("z0", &cfd::Grid3D::z0)
        .def("total_cells", &cfd::Grid3D::total_cells);

    py::class_<cfd::FieldStats>(m, "FieldStats")
        .def_readonly("min_val", &cfd::FieldStats::min_val)
        .def_readonly("max_val", &cfd::FieldStats::max_val)
        .def_readonly("sum_val", &cfd::FieldStats::sum_val)
        .def_readonly("mean_val", &cfd::FieldStats::mean_val)
        .def_readonly("rms_val", &cfd::FieldStats::rms_val);

    // Velocity magnitude
    m.def("velocity_magnitude", [](py::array_t<double, py::array::c_style | py::array::forcecast> u_arr,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> v_arr,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
                                   int threads, bool use_simd) {
        py::buffer_info u_info = u_arr.request();
        py::buffer_info v_info = v_arr.request();
        py::buffer_info w_info = w_arr.request();

        if (u_info.size != v_info.size || u_info.size != w_info.size) {
            throw std::runtime_error("Velocity array size mismatch.");
        }

        const size_t n = u_info.size;
        auto mag_arr = py::array_t<double>(u_info.shape);
        py::buffer_info mag_info = mag_arr.request();

        const double* u_ptr = static_cast<const double*>(u_info.ptr);
        const double* v_ptr = static_cast<const double*>(v_info.ptr);
        const double* w_ptr = static_cast<const double*>(w_info.ptr);
        double* mag_ptr = static_cast<double*>(mag_info.ptr);

        {
            py::gil_scoped_release release;
            if (threads == 1) {
                cfd::velocity_magnitude_serial(u_ptr, v_ptr, w_ptr, mag_ptr, n);
            } else if (use_simd) {
                cfd::velocity_magnitude_simd(u_ptr, v_ptr, w_ptr, mag_ptr, n, threads);
            } else {
                cfd::velocity_magnitude_openmp(u_ptr, v_ptr, w_ptr, mag_ptr, n, threads);
            }
        }

        return mag_arr;
    }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("threads") = 0, py::arg("use_simd") = false,
       "Calculate velocity magnitude using serial, OpenMP, or SIMD kernels.");

    // Kinetic energy
    m.def("kinetic_energy", [](py::array_t<double, py::array::c_style | py::array::forcecast> u_arr,
                               py::array_t<double, py::array::c_style | py::array::forcecast> v_arr,
                               py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
                               int threads) {
        py::buffer_info u_info = u_arr.request();
        py::buffer_info v_info = v_arr.request();
        py::buffer_info w_info = w_arr.request();

        const size_t n = u_info.size;
        const double* u_ptr = static_cast<const double*>(u_info.ptr);
        const double* v_ptr = static_cast<const double*>(v_info.ptr);
        const double* w_ptr = static_cast<const double*>(w_info.ptr);

        double ke = 0.0;
        {
            py::gil_scoped_release release;
            if (threads == 1) {
                ke = cfd::kinetic_energy_serial(u_ptr, v_ptr, w_ptr, n);
            } else {
                ke = cfd::kinetic_energy_openmp(u_ptr, v_ptr, w_ptr, n, threads);
            }
        }
        return ke;
    }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("threads") = 0,
       "Calculate total kinetic energy across the field.");

    // Field statistics
    m.def("field_stats", [](py::array_t<double, py::array::c_style | py::array::forcecast> data_arr,
                            int threads) {
        py::buffer_info info = data_arr.request();
        const size_t n = info.size;
        const double* ptr = static_cast<const double*>(info.ptr);

        cfd::FieldStats stats;
        {
            py::gil_scoped_release release;
            if (threads == 1) {
                stats = cfd::field_stats_serial(ptr, n);
            } else {
                stats = cfd::field_stats_openmp(ptr, n, threads);
            }
        }
        return stats;
    }, py::arg("data"), py::arg("threads") = 0,
       "Calculate min, max, sum, mean, and RMS of a scalar field.");

    // Gradients
    m.def("gradients", [](py::array_t<double, py::array::c_style | py::array::forcecast> phi_arr,
                          size_t nx, size_t ny, size_t nz,
                          double dx, double dy, double dz,
                          int collapse_level, int threads) {
        py::buffer_info info = phi_arr.request();
        if (info.size != static_cast<ssize_t>(nx * ny * nz)) {
            throw std::runtime_error("Field dimensions do not match nx * ny * nz.");
        }

        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        auto gx_arr = py::array_t<double>({nx, ny, nz});
        auto gy_arr = py::array_t<double>({nx, ny, nz});
        auto gz_arr = py::array_t<double>({nx, ny, nz});

        const double* phi_ptr = static_cast<const double*>(info.ptr);
        double* gx_ptr = static_cast<double*>(gx_arr.request().ptr);
        double* gy_ptr = static_cast<double*>(gy_arr.request().ptr);
        double* gz_ptr = static_cast<double*>(gz_arr.request().ptr);

        {
            py::gil_scoped_release release;
            if (threads == 1) {
                cfd::gradient_serial(phi_ptr, grid, gx_ptr, gy_ptr, gz_ptr);
            } else {
                cfd::gradient_openmp(phi_ptr, grid, gx_ptr, gy_ptr, gz_ptr, collapse_level, threads);
            }
        }

        return py::make_tuple(gx_arr, gy_arr, gz_arr);
    }, py::arg("phi"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("collapse_level") = 2, py::arg("threads") = 0,
       "Calculate finite-difference gradients (dphi/dx, dphi/dy, dphi/dz).");

    // Divergence
    m.def("divergence", [](py::array_t<double, py::array::c_style | py::array::forcecast> u_arr,
                           py::array_t<double, py::array::c_style | py::array::forcecast> v_arr,
                           py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
                           size_t nx, size_t ny, size_t nz,
                           double dx, double dy, double dz,
                           int collapse_level, int threads) {
        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        auto div_arr = py::array_t<double>({nx, ny, nz});

        const double* u_ptr = static_cast<const double*>(u_arr.request().ptr);
        const double* v_ptr = static_cast<const double*>(v_arr.request().ptr);
        const double* w_ptr = static_cast<const double*>(w_arr.request().ptr);
        double* div_ptr = static_cast<double*>(div_arr.request().ptr);

        {
            py::gil_scoped_release release;
            if (threads == 1) {
                cfd::divergence_serial(u_ptr, v_ptr, w_ptr, grid, div_ptr);
            } else {
                cfd::divergence_openmp(u_ptr, v_ptr, w_ptr, grid, div_ptr, collapse_level, threads);
            }
        }

        return div_arr;
    }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("collapse_level") = 2, py::arg("threads") = 0,
       "Calculate velocity field divergence div = du/dx + dv/dy + dw/dz.");

    // Vorticity
    m.def("vorticity", [](py::array_t<double, py::array::c_style | py::array::forcecast> u_arr,
                          py::array_t<double, py::array::c_style | py::array::forcecast> v_arr,
                          py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
                          size_t nx, size_t ny, size_t nz,
                          double dx, double dy, double dz,
                          bool compute_magnitude, int collapse_level, int threads) -> py::tuple {
        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        auto wx_arr = py::array_t<double>({nx, ny, nz});
        auto wy_arr = py::array_t<double>({nx, ny, nz});
        auto wz_arr = py::array_t<double>({nx, ny, nz});
        auto mag_arr = compute_magnitude ? py::array_t<double>({nx, ny, nz}) : py::array_t<double>();

        const double* u_ptr = static_cast<const double*>(u_arr.request().ptr);
        const double* v_ptr = static_cast<const double*>(v_arr.request().ptr);
        const double* w_ptr = static_cast<const double*>(w_arr.request().ptr);
        double* wx_ptr = static_cast<double*>(wx_arr.request().ptr);
        double* wy_ptr = static_cast<double*>(wy_arr.request().ptr);
        double* wz_ptr = static_cast<double*>(wz_arr.request().ptr);
        double* mag_ptr = compute_magnitude ? static_cast<double*>(mag_arr.request().ptr) : nullptr;

        {
            py::gil_scoped_release release;
            if (threads == 1) {
                cfd::vorticity_serial(u_ptr, v_ptr, w_ptr, grid, wx_ptr, wy_ptr, wz_ptr, mag_ptr);
            } else {
                cfd::vorticity_openmp(u_ptr, v_ptr, w_ptr, grid, wx_ptr, wy_ptr, wz_ptr, mag_ptr,
                                      collapse_level, threads);
            }
        }

        if (compute_magnitude) {
            return py::make_tuple(wx_arr, wy_arr, wz_arr, mag_arr);
        } else {
            return py::make_tuple(wx_arr, wy_arr, wz_arr, py::none());
        }
    }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("compute_magnitude") = true, py::arg("collapse_level") = 2, py::arg("threads") = 0,
       "Calculate vorticity vector (wx, wy, wz) and magnitude.");

    // Q-criterion
    m.def("q_criterion", [](py::array_t<double, py::array::c_style | py::array::forcecast> u_arr,
                            py::array_t<double, py::array::c_style | py::array::forcecast> v_arr,
                            py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
                            size_t nx, size_t ny, size_t nz,
                            double dx, double dy, double dz,
                            int collapse_level, int threads) {
        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        auto q_arr = py::array_t<double>({nx, ny, nz});

        const double* u_ptr = static_cast<const double*>(u_arr.request().ptr);
        const double* v_ptr = static_cast<const double*>(v_arr.request().ptr);
        const double* w_ptr = static_cast<const double*>(w_arr.request().ptr);
        double* q_ptr = static_cast<double*>(q_arr.request().ptr);

        {
            py::gil_scoped_release release;
            if (threads == 1) {
                cfd::q_criterion_serial(u_ptr, v_ptr, w_ptr, grid, q_ptr);
            } else {
                cfd::q_criterion_openmp(u_ptr, v_ptr, w_ptr, grid, q_ptr, collapse_level, threads);
            }
        }

        return q_arr;
    }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("collapse_level") = 2, py::arg("threads") = 0,
       "Calculate Q-criterion vortex identification field Q = 0.5 * (||Omega||^2 - ||S||^2).");

    // Synthetic field generation
    m.def("generate_taylor_green", [](size_t nx, size_t ny, size_t nz,
                                      double dx, double dy, double dz,
                                      double U0, double L, double rho, double p0) {
        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        cfd::Field3D field = cfd::generate_taylor_green_vortex(grid, U0, L, rho, p0);

        py::dict res;
        res["u"] = py::array_t<double>({nx, ny, nz}, field.u.data());
        res["v"] = py::array_t<double>({nx, ny, nz}, field.v.data());
        res["w"] = py::array_t<double>({nx, ny, nz}, field.w.data());
        res["pressure"] = py::array_t<double>({nx, ny, nz}, field.pressure.data());
        return res;
    }, py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("U0") = 1.0, py::arg("L") = 1.0, py::arg("rho") = 1.0, py::arg("p0") = 100.0,
       "Generate 3D Taylor-Green vortex analytical velocity and pressure fields.");

    m.def("generate_solid_body", [](size_t nx, size_t ny, size_t nz,
                                    double dx, double dy, double dz,
                                    double Omega, double rho, double p0) {
        cfd::Grid3D grid(nx, ny, nz, dx, dy, dz);
        cfd::Field3D field = cfd::generate_solid_body_rotation(grid, Omega, rho, p0);

        py::dict res;
        res["u"] = py::array_t<double>({nx, ny, nz}, field.u.data());
        res["v"] = py::array_t<double>({nx, ny, nz}, field.v.data());
        res["w"] = py::array_t<double>({nx, ny, nz}, field.w.data());
        res["pressure"] = py::array_t<double>({nx, ny, nz}, field.pressure.data());
        return res;
    }, py::arg("nx"), py::arg("ny"), py::arg("nz"),
       py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0,
       py::arg("Omega") = 2.0, py::arg("rho") = 1.0, py::arg("p0") = 100.0,
       "Generate solid body rotation analytical velocity and pressure fields.");

    // Experiments
    m.def("run_false_sharing_experiment", [](size_t iterations, int threads) {
        cfd::FalseSharingResult res = cfd::run_false_sharing_experiment(iterations, threads);
        py::dict d;
        d["unpadded_seconds"] = res.unpadded_seconds;
        d["padded_seconds"] = res.padded_seconds;
        d["speedup"] = res.speedup_from_padding;
        d["threads"] = res.num_threads;
        d["iterations"] = res.iterations;
        return d;
    }, py::arg("iterations") = 50000000, py::arg("threads") = 0);

    m.def("run_race_condition_experiment", [](int64_t iterations_per_thread, int threads) {
        auto results = cfd::run_race_condition_experiment(iterations_per_thread, threads);
        py::list list_res;
        for (const auto& r : results) {
            py::dict d;
            d["method"] = r.method;
            d["expected"] = r.expected_sum;
            d["actual"] = r.actual_sum;
            d["error"] = r.error;
            d["elapsed"] = r.elapsed_seconds;
            list_res.append(d);
        }
        return list_res;
    }, py::arg("iterations_per_thread") = 5000000, py::arg("threads") = 0);

    m.def("test_floating_point_nonassociativity", [](size_t n, int threads) {
        auto res = cfd::test_floating_point_nonassociativity(n, threads);
        py::dict d;
        d["serial_sum"] = res.serial_sum;
        d["parallel_sum"] = res.parallel_sum;
        d["abs_diff"] = res.abs_diff;
        d["rel_diff"] = res.rel_diff;
        return d;
    }, py::arg("n") = 10000000, py::arg("threads") = 0);
}
