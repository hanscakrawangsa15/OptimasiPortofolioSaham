# Panduan kerja bertiga

## Cabang dan alur

- `main` hanya berisi hasil yang sudah lulus tes dan di-review. Jangan commit langsung ke `main`.
- Tiap orang bekerja di cabang pribadinya (`Hans_130`, `Antika_034`, `Fika_229`). Untuk pekerjaan besar,
  buat cabang turunan `Nama_fitur`, mis. `Hans_130_decoder`.
- Alur: kerjakan di cabang -> `git pull origin main` (gabungkan main ke cabangmu) -> `python3 -m pytest tests -q`
  hijau -> push -> buka Pull Request ke `main` -> satu orang lain me-review -> merge.
- Jangan `git push --force` ke cabang yang dipakai orang lain.

## Pemilik modul

Satu modul punya satu pemilik pada satu waktu (lihat tabel PIC di README). Mengubah modul orang lain: kabari
dulu atau lewat PR kecil yang direview pemiliknya.

## Antarmuka

Tanda tangan fungsi dan kelas ada di spesifikasi bagian 5. Mengubah antarmuka (`Problem`, `Result`,
`run(problem, budget, seed)`, `Evaluator`, dekoder) harus lewat PR yang juga memperbarui spesifikasi, karena
orang lain bergantung padanya.

## Commit

- Pesan commit bahasa Indonesia dengan awalan: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `data:`.
- Satu commit satu perubahan logis. Tes ikut dalam commit yang sama dengan kodenya.
- Kode baru dengan TDD: tulis tes, lihat gagal, lalu implementasikan.

## Kode

- Acak hanya lewat `numpy.random.default_rng(seed)`, jangan `np.random` global. Seed sama harus memberi hasil
  identik.
- Semua parameter masalah dan algoritma ada di `config.py`, bukan angka tersebar di kode.
- Semua metaheuristik memakai `Evaluator` yang sama untuk menghitung evaluasi, supaya anggaran 20.000 evaluasi
  adil. Jangan menghitung fitness di luar `Evaluator`.
- Algoritma hanya menerima `Problem` (data latih). Jangan membaca return uji di dalam algoritma.
- Return selalu return sederhana (bukan log). Lot dihitung dari harga raw, return dari Adj Close.

## Data

- Hanya satu orang yang mengunduh ulang data (`python3 src/scrappingdata.py`) dan meng-commit hasilnya dengan
  pesan `data:`, karena yfinance bisa mengubah nilai historis. Yang lain menarik lewat `git pull`.
- Jangan meng-commit file besar di luar `data/` dan `hasil/`. PDF di `docs/rujukan/` sebaiknya tidak ikut
  di-push kecuali disepakati.
- Keluaran eksperimen ke `hasil/csv/` dan `hasil/gambar/`.
