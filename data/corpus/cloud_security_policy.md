# Kebijakan Keamanan Cloud & Manajemen Rahasia (Cloud Security Policy)

## Prinsip Zero Trust dan Least Privilege (RBAC)
Seluruh infrastruktur cloud produksi wajib menerapkan arsitektur **Zero Trust** dan prinsip hak akses minimal (**Least Privilege**). Pengaturan akses berbasis peran (**Role-Based Access Control / RBAC**) harus memenuhi ketentuan berikut:
- Penggunaan akun root atau super-admin untuk operasional harian dilarang keras; akun root wajib diamankan dengan Hardware MFA (FIDO2/YubiKey) dan disimpan dalam brankas break-glass.
- Akses ke lingkungan produksi menggunakan mekanisme **Just-in-Time (JIT)** dengan durasi maksimal **4 jam** dan persetujuan rekan kerja (peer approval).
- Peninjauan ulang hak akses (access recertification) wajib dilakukan setiap **90 hari (kuartalan)** untuk seluruh engineer, service account, dan integrasi pihak ketiga.

## Enkripsi Data dan Key Management Service (KMS)
Seluruh data perusahaan, termasuk database vektor, indeks pencarian, dan penyimpanan objek (object storage), wajib dienkripsi baik saat istirahat maupun saat transit:
- **Encryption at Rest**: Menggunakan standar **AES-256-GCM** yang dikelola melalui Customer-Managed Keys (CMK) di **Key Management Service (KMS)** dengan dukungan Hardware Security Module (HSM).
- **Encryption in Transit**: Seluruh komunikasi eksternal dan antar-layanan (service-to-service) wajib menggunakan **TLS 1.3** (minimal TLS 1.2 untuk integrasi legacy yang telah mendapat pengecualian tertulis) dengan Mutual TLS (mTLS) di dalam cluster produksi.
- **Envelope Encryption**: Kunci enkripsi data (DEK) dienkripsi menggunakan Key Encryption Key (KEK) di KMS sehingga rotasi kunci utama dapat berjalan tanpa mendekripsi ulang seluruh volume penyimpanan secara manual.

## Manajemen Rahasia (Secrets Management) dan Kebijakan Rotasi
Menyimpan kredensial, API key LLM (OpenAI/Gemini), password database, atau private key secara plaintext di dalam kode sumber Git, file konfigurasi, atau image Docker adalah pelanggaran berat.
- Seluruh rahasia wajib disimpan di centralized Secrets Manager (seperti HashiCorp Vault atau Cloud Secret Manager) dan disuntikkan saat runtime melalui variabel lingkungan terenkripsi atau ephemeral volume mount.
- **Jadwal Rotasi Rahasia Wajib**:
  - API Key layanan eksternal dan token CI/CD: rotasi otomatis setiap **90 hari**.
  - Kredensial database produksi dan service account key: rotasi otomatis setiap **30 hari** (atau menggunakan dynamic short-lived credentials berdurasi 1 jam).
  - Kunci enkripsi utama KMS (CMK): rotasi tahunan otomatis setiap **365 hari**.
- Setiap repositori Git wajib mengaktifkan **pre-commit secret scanning** dan CI secret detection; apabila rahasia terlanjur terpublikasi di commit Git, rahasia tersebut wajib dicabut (revoke) dalam waktu **15 menit** dan dirotasi penuh.

## Audit Logging dan Retensi Log Keamanan
Semua aktivitas autentikasi, perubahan kebijakan IAM/RBAC, akses KMS, pengambilan rahasia, dan pemanggilan API sensitif wajib dicatat dalam **immutable audit log** (append-only, dilindungi Object Lock / WORM):
- Log keamanan harus menyertakan timestamp UTC, actor identity, source IP, action name, target resource, dan status hasil (allow/deny), tanpa pernah mencatat nilai rahasia (secret value) atau PII mentah di dalam log.
- **Masa Retensi Log**: Log operasional panas (hot storage) disimpan minimal **90 hari** di SIEM untuk deteksi anomali real-time, sedangkan arsip dingin (cold immutable archive) disimpan minimal **3 tahun** (atau **5 tahun** untuk sistem keuangan dan kepatuhan regulasi).

## Segmentasi Jaringan dan Pertahanan Container
Layanan AI dan database indeks berjalan di dalam private subnet tanpa alamat IP publik langsung. Trafik masuk hanya melalui API Gateway / WAF dengan rate limiting aktif. Image Docker produksi wajib dibangun dari base image minimal (slim/distroless), berjalan sebagai **non-root user**, serta lulus pemindaian kerentanan (vulnerability scan) tanpa celah **Critical** atau **High** yang belum dimitigasi sebelum deployment.
