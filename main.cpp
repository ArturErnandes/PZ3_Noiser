#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int clampi(int v, int lo, int hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }

void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (a[j] > a[j + 1]) { int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
}

bool read_pgm(const string& file_name, int**& img, int& w, int& h, int& maxv) {
    ifstream f(file_name);
    if (!f.is_open()) return false;

    string fmt; f >> fmt;
    if (fmt != "P2") return false;

    while (f.peek() == '#' || f.peek() == '\n' || f.peek() == '\r')
        f.ignore(1000, '\n');

    f >> w >> h >> maxv;

    img = new int*[h];
    for (int i = 0; i < h; i++) img[i] = new int[w];

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            f >> img[i][j];

    f.close();
    return true;
}

void save_pgm(const string& file_name, int** img, int w, int h, int maxv) {
    ofstream f(file_name);
    f << "P2\n" << w << " " << h << "\n" << maxv << "\n";
    int c = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            f << img[i][j];
            if (++c % 17 == 0) f << "\n"; else f << " ";
        }
    }
    f << "\n";
    f.close();
}

double mse_val(int** a, int** b, int h, int w) {
    double s = 0.0, n = (double)h * (double)w;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            double d = (double)a[i][j] - (double)b[i][j];
            s += d * d;
        }
    return s / n;
}
double psnr_val(int** a, int** b, int h, int w, int maxv) {
    double m = mse_val(a, b, h, w);
    if (m <= 0.0) return 1e9;
    double L = (double)maxv;
    return 10.0 * log10((L * L) / m);
}

double randn01() { 
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void noise_sp(int** src, int** dst, int h, int w, int maxv, double p) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < p * 0.5) dst[i][j] = 0;
            else if (r < p)  dst[i][j] = maxv;
            else             dst[i][j] = src[i][j];
        }
}

void noise_gauss(int** src, int** dst, int h, int w, int maxv, double sigma) {
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int v = src[i][j] + (int)lround(randn01() * sigma);
            dst[i][j] = clampi(v, 0, maxv);
        }
}

// ---------- фильтры ----------
void filt_box(int k, int h, int w, int** in, int** tmp, int maxv) {
    if (k % 2 == 0) k += 1;
    int r = k / 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int sum = 0, cnt = 0;
            for (int dy = -r; dy <= r; dy++)
                for (int dx = -r; dx <= r; dx++) {
                    int y = i + dy, x = j + dx;
                    if (y >= 0 && y < h && x >= 0 && x < w) { sum += in[y][x]; cnt++; }
                }
            tmp[i][j] = clampi(cnt ? sum / cnt : in[i][j], 0, maxv);
        }
    }
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            in[i][j] = tmp[i][j];
}

void filt_median(int k, int h, int w, int** in, int** tmp) {
    if (k % 2 == 0) k += 1;
    int r = k / 2;
    int n = k * k;
    int* arr = new int[n];

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int t = 0;
            for (int dy = -r; dy <= r; dy++)
                for (int dx = -r; dx <= r; dx++) {
                    int y = i + dy, x = j + dx;
                    if (y >= 0 && y < h && x >= 0 && x < w) arr[t++] = in[y][x];
                }
            bubble_sort(arr, t);
            tmp[i][j] = arr[t / 2];
        }
    }
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            in[i][j] = tmp[i][j];

    delete[] arr;
}

void filt_ema2d(double k, int h, int w, int** in, int maxv) {
    double* buf = new double[h * w];
    for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) buf[i * w + j] = (double)in[i][j];

    for (int i = 0; i < h; i++) {
        double yv = buf[i * w + 0];
        for (int j = 1; j < w; j++) { double x = buf[i * w + j]; yv = yv + (x - yv) * k; buf[i * w + j] = yv; }
    }
    for (int i = 0; i < h; i++) {
        double yv = buf[i * w + (w - 1)];
        for (int j = w - 2; j >= 0; j--) { double x = buf[i * w + j]; yv = yv + (x - yv) * k; buf[i * w + j] = 0.5 * (buf[i * w + j] + yv); }
    }
    for (int j = 0; j < w; j++) {
        double yv = buf[0 * w + j];
        for (int i = 1; i < h; i++) { double x = buf[i * w + j]; yv = yv + (x - yv) * k; buf[i * w + j] = yv; }
    }
    for (int j = 0; j < w; j++) {
        double yv = buf[(h - 1) * w + j];
        for (int i = h - 2; i >= 0; i--) { double x = buf[i * w + j]; yv = yv + (x - yv) * k; buf[i * w + j] = 0.5 * (buf[i * w + j] + yv); }
    }

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int v = (int)lround(buf[i * w + j]);
            in[i][j] = clampi(v, 0, maxv);
        }

    delete[] buf;
}

void make_dirs() {
    fs::create_directories("images");
    fs::create_directories("images_noise");
    fs::create_directories("images_denoise");
    fs::create_directories("results");
}
void csv_header(const string& path) {
    bool need = !fs::exists(path);
    ofstream f(path, ios::app);
    if (need) f << "Файл,Шум,Параметр,Фильтр,MSE,PSNR\n";
}

int main() {
    srand((unsigned)time(NULL));
    make_dirs();

    const string dir_in  = "images";
    const string dir_nz  = "images_noise";
    const string dir_dn  = "images_denoise";
    const string csvp    = "results/results.csv";
    csv_header(csvp);
    ofstream csv(csvp, ios::app);

    bool   use_gauss = true;   
    double sigma     = 20.0;   
    double prob      = 0.03; 
    int    box_k     = 3;    
    int    med_k     = 3;    
    double ema_k     = 0.5; 

    for (auto& e : fs::directory_iterator(dir_in)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        if (p.extension() != ".pgm") continue;

        int w = 0, h = 0, maxv = 0;
        int** img = nullptr;
        if (!read_pgm(p.string(), img, w, h, maxv)) {
            cerr << "Не удалось прочитать " << p << "\n";
            continue;
        }
        string base = p.stem().string();

        int** noisy = new int*[h];
        int** work  = new int*[h];
        int** tmp   = new int*[h];
        for (int i = 0; i < h; i++) { noisy[i] = new int[w]; work[i] = new int[w]; tmp[i] = new int[w]; }

        string noise_tag;
        if (use_gauss) { noise_gauss(img, noisy, h, w, maxv, sigma); noise_tag = "gauss_" + to_string((int)sigma); }
        else           { noise_sp(img, noisy, h, w, maxv, prob);     noise_tag = "sp_" + to_string((int)(prob * 100)); }

        save_pgm( (fs::path(dir_nz) / (base + "__" + noise_tag + ".pgm")).string(), noisy, w, h, maxv );


        for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) work[i][j] = noisy[i][j];
        filt_box(box_k, h, w, work, tmp, maxv);
        save_pgm( (fs::path(dir_dn) / (base + "__" + noise_tag + "__box" + to_string(box_k) + ".pgm")).string(), work, w, h, maxv );
        csv << base << "," << (use_gauss ? "gaussian" : "salt_pepper") << ","
            << (use_gauss ? to_string(sigma) : to_string(prob)) << ","
            << "box" << box_k << "," << mse_val(img, work, h, w) << "," << psnr_val(img, work, h, w, maxv) << "\n";

        for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) work[i][j] = noisy[i][j];
        filt_median(med_k, h, w, work, tmp);
        save_pgm( (fs::path(dir_dn) / (base + "__" + noise_tag + "__med" + to_string(med_k) + ".pgm")).string(), work, w, h, maxv );
        csv << base << "," << (use_gauss ? "gaussian" : "salt_pepper") << ","
            << (use_gauss ? to_string(sigma) : to_string(prob)) << ","
            << "median" << med_k << "," << mse_val(img, work, h, w) << "," << psnr_val(img, work, h, w, maxv) << "\n";

        for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) work[i][j] = noisy[i][j];
        filt_ema2d(ema_k, h, w, work, maxv);
        save_pgm( (fs::path(dir_dn) / (base + "__" + noise_tag + "__ema.pgm")).string(), work, w, h, maxv );
        csv << base << "," << (use_gauss ? "gaussian" : "salt_pepper") << ","
            << (use_gauss ? to_string(sigma) : to_string(prob)) << ","
            << "ema(k=0.5)" << "," << mse_val(img, work, h, w) << "," << psnr_val(img, work, h, w, maxv) << "\n";

        for (int i = 0; i < h; i++) { delete[] img[i]; delete[] noisy[i]; delete[] work[i]; delete[] tmp[i]; }
        delete[] img; delete[] noisy; delete[] work; delete[] tmp;
    }

    cout << "Готово. Шум: images_noise/, фильтры: images_denoise/, метрики: results/results.csv\n";
    return 0;
}