# Preparing your rig

From bare hardware to a serving Qwen3.8-Flash-Next on two Radeon AI PRO R9700s: firmware, Arch Linux,
Docker, peer-to-peer between the cards, power tuning, compose. Each step ends with a check.

The reference rig: MSI MPG X670E CARBON WIFI, Ryzen 9 7950X3D, 64 GB RAM, one 2 TB NVMe, two R9700s
in the two CPU-attached x16 slots. Another board needs the same settings under its own menu names.

## 1. Firmware setup

Enter the firmware setup (Del at power-on) and set:

| Setting | Value | Why |
|---|---|---|
| XMP / EXPO | profile 1 | the RAM runs at its rated speed instead of JEDEC defaults |
| Above 4G Decoding | enabled | needed for Resizable BAR |
| Re-Size BAR Support | enabled | the host writes each dispatch's arguments straight into VRAM, and peer access needs the large BAR |
| CSM | disabled | Resizable BAR needs pure UEFI boot |
| Secure Boot | disabled | the Arch install medium is not signed |
| PCIe lane split of the two x16 slots | x8/x8 | both cards on CPU lanes |
| PCIe link speed of both slots | Gen5 | stops the slots training at a lower generation |
| Onboard Wi-Fi, Bluetooth | disabled | the rig is wired, and an unused device is one less driver and IRQ |

Save and exit (F10).

## 2. Install Arch Linux

Boot the [Arch ISO](https://archlinux.org/download/) from USB with the network cable plugged in, then
run `archinstall` with these choices. Leave everything else at its default.

| archinstall item | Choice |
|---|---|
| Disk configuration | best-effort default layout, ext4 |
| Swap | off |
| Bootloader | systemd-boot |
| User account | your user, as superuser (sudo) |
| Profile | Minimal (no desktop environment) |
| Audio | none |
| Kernels | linux |
| Network configuration | NetworkManager |
| Additional packages | `openssh git base-devel` |

Install, reboot into the new system, log in on the console and start the SSH server:

```sh
sudo systemctl enable --now sshd
ip -4 addr show      # note the rig's address
```

Go back to your PC. Everything from here on runs over SSH:

```sh
ssh-copy-id you@RIG_ADDRESS
ssh you@RIG_ADDRESS
```

## 3. Docker and the AUR helper

```sh
sudo pacman -Syu --needed docker docker-compose docker-buildx
sudo systemctl enable --now docker
sudo usermod -aG docker,render,video "$USER"     # log out and back in

git clone https://aur.archlinux.org/yay.git && (cd yay && makepkg -si) && rm -rf yay
yay -S rocm-bandwidth-test
```

## 4. Kernel parameters

Edit the boot entry archinstall wrote (`/boot/loader/entries/*_linux.conf`) and append these to its
`options` line:

```
iommu=pt amdgpu.ppfeaturemask=0xfff7ffff amdgpu.ras_enable=0
```

- `iommu=pt` keeps the IOMMU in passthrough, so the cards can reach each other's memory and pinned
  host memory maps without translation.
- `amdgpu.ppfeaturemask=0xfff7ffff` is the driver's default mask with the OverDrive bit set. OverDrive
  is what makes the undervolt in step 6 writable.
- `amdgpu.ras_enable=0` turns VRAM ECC off. With ECC on, each card reserves about 2 GB of its VRAM.

Arch's stock kernel has enabled peer-to-peer (`CONFIG_HSA_AMD_P2P`) since 7.1.2, so a current install
needs no custom kernel. An older one would need a rebuild with that option. Make the entry the default
and reboot:

```sh
bootctl list                                  # the linux entry, with the new options
sudo bootctl set-default "$(basename /boot/loader/entries/*_linux.conf)"
sudo reboot
```

## 5. Check the cards

```sh
cat /proc/cmdline                                                # the three parameters
zcat /proc/config.gz | grep -E 'HSA_AMD_P2P|PCI_P2PDMA'          # both =y
lspci -vv -d 1002:7551 | grep 'Region 0'                         # size=32G on both cards (ReBAR)
sudo journalctl -k -b | grep 'available PCIe bandwidth'          # 32.0 GT/s PCIe x8 per card
cat /sys/class/drm/card*/device/mem_info_vram_total              # 34208743424 (32624 MiB) on both: ECC off
rocm-bandwidth-test -a
```

In the bandwidth matrices, each card reaches about 28 GB/s to and from the CPU. Every GPU-to-GPU cell
must show a rate. A missing cell means no peer access: check `iommu=pt` and the BAR size.

An idle link reads 2.5 GT/s in `lspci`, and that is normal. If one card measures half the other's rate
under load, the PCIe power state of its root port is stuck one level down, and a reboot clears it.

## 6. Power limit and undervolt

210 W costs under 1% of decode speed against the 300 W default. Use -70 mV: at -75 mV the reference
rig's heaviest GEMMs gave run-to-run differences. Both settings are lost at every reboot, and the
undervolt also at every runtime resume of the card. A boot service applies them and keeps the cards
awake:

```sh
sudo tee /usr/local/sbin/gpu-tune.sh >/dev/null <<'EOF'
#!/bin/bash
# Undervolt and power-cap every R9700, then return the cards to automatic clocks.
# Needs amdgpu.ppfeaturemask=0xfff7ffff on the kernel command line (OverDrive).
set -u
OFFSET='vo -70'
CAP_UW=210000000
rc=0
for d in /sys/bus/pci/drivers/amdgpu/0000:*; do
  [ "$(cat "$d/device")" = 0x7551 ] || continue
  for _ in $(seq 1 30); do [ -e "$d/pp_od_clk_voltage" ] && break; sleep 1; done
  [ -e "$d/pp_od_clk_voltage" ] || { echo "${d##*/}: no OverDrive interface" >&2; rc=1; continue; }
  echo on        > "$d/power/control"      # a runtime resume drops the OverDrive table
  echo manual    > "$d/power_dpm_force_performance_level"
  echo "$OFFSET" > "$d/pp_od_clk_voltage"
  echo c         > "$d/pp_od_clk_voltage"
  echo auto      > "$d/power_dpm_force_performance_level"
  for h in "$d"/hwmon/hwmon*; do echo "$CAP_UW" > "$h/power1_cap"; done
  echo "${d##*/}: $(grep -A1 OD_VDDGFX "$d/pp_od_clk_voltage" | tail -1)," \
       "cap $(( $(cat "$d"/hwmon/hwmon*/power1_cap) / 1000000 )) W"
done
exit $rc
EOF
sudo chmod +x /usr/local/sbin/gpu-tune.sh

sudo tee /etc/systemd/system/gpu-tune.service >/dev/null <<'EOF'
[Unit]
Description=R9700 undervolt and power cap
After=multi-user.target
ConditionKernelCommandLine=amdgpu.ppfeaturemask=0xfff7ffff

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/gpu-tune.sh
RemainAfterExit=yes
TimeoutStartSec=90

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now gpu-tune.service
journalctl -u gpu-tune -b --no-pager | tail -2       # -70mV, cap 210 W on both cards
```

## 7. Serve Qwen3.8-Flash-Next

```sh
git clone https://codeberg.org/StillDeadcode/radiance.git
cp -r radiance/deploy/compose ~/radiance-compose && cd ~/radiance-compose

mkdir -p ~/models/rad
curl -L -C - -o ~/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad \
  https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/resolve/main/qwen3.8-next-flash-fp8-iq4r-moe.rad
                                               # 114 GiB; -C - resumes after an interruption

cp .env.example .env && chmod 600 .env
```

Set these in `.env`:

| Variable | Value |
|---|---|
| `RADIANCE_MODELS` | `/home/you/models/rad` |
| `RADIANCE_API_KEY` | a long random string (`openssl rand -hex 32`) |
| `RADIANCE_UID`, `RADIANCE_GID` | `id -u`, `id -g` |
| `RENDER_GID`, `VIDEO_GID` | `getent group render \| cut -d: -f3`, the same for `video` |

```sh
docker compose -f flashnext.yaml up -d
docker compose -f flashnext.yaml logs -f       # Ctrl-C once it serves; the container keeps running
```

The startup log must say `device: the 2 cards reach each other's memory directly (peer to peer)`. A
`no peer-to-peer access` warning means step 5's checks did not all pass. Then:

```sh
KEY=$(grep ^RADIANCE_API_KEY= .env | cut -d= -f2)
curl -s localhost:8000/health
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 65 * 3?"}]}'
```

The container restarts with the machine. `docker compose -f flashnext.yaml down` stops it.
[`DOCKER.md`](DOCKER.md) lists the other models' compose files, and [`GUIDE.md`](GUIDE.md) covers
serving flags and the API.
