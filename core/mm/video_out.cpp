/* video_out.cpp -- VideoWriter (video_out.h) over libavformat, libavcodec, libswscale and
 * libswresample. */
#include "mm/video_out.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#ifdef RAD_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace rad {
namespace mm {

#ifndef RAD_HAVE_FFMPEG

struct VideoWriter::Impl {};
VideoWriter::VideoWriter() = default;
VideoWriter::~VideoWriter() = default;
static int no_ffmpeg(std::string* why) {
    if (why) *why = "this build has no libavcodec, so it writes no video";
    return RAD_E_UNSUPPORTED;
}
int VideoWriter::open(int32_t, int32_t, int64_t, int64_t, int64_t, int64_t, std::string* why,
                      VideoFormat) {
    return no_ffmpeg(why);
}
int VideoWriter::add_frames(const uint8_t*, int32_t, std::string* why) { return no_ffmpeg(why); }
int VideoWriter::add_audio(const float*, int64_t, std::string* why) { return no_ffmpeg(why); }
int VideoWriter::finish(std::string*, std::string* why) { return no_ffmpeg(why); }

#else

namespace {

/* The file in memory. The MP4 muxer seeks back to patch sizes and, to put the moov atom first,
 * reopens its output for reading and moves the media data up (FFmpeg's shift_data): the reader is
 * a second handle on the same bytes with a position of its own. The WebM muxer only seeks within
 * this one handle, so it never needs the reader. */
struct MemFile {
    std::string data;
    int64_t     pos = 0;
};
struct MemReader {
    MemFile* f = nullptr;
    int64_t  pos = 0;
};

int mem_write(void* op, const uint8_t* buf, int n) {
    MemFile* f = static_cast<MemFile*>(op);
    const int64_t end = f->pos + n;
    if (end > (int64_t)f->data.size()) f->data.resize((size_t)end);
    std::memcpy(&f->data[(size_t)f->pos], buf, (size_t)n);
    f->pos = end;
    return n;
}

int64_t mem_seek(void* op, int64_t off, int whence) {
    MemFile* f = static_cast<MemFile*>(op);
    const int64_t size = (int64_t)f->data.size();
    switch (whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE: return size;
    case SEEK_SET: f->pos = off; break;
    case SEEK_CUR: f->pos += off; break;
    case SEEK_END: f->pos = size + off; break;
    default: return AVERROR(EINVAL);
    }
    return f->pos < 0 ? AVERROR(EINVAL) : f->pos;
}

int mem_read(void* op, uint8_t* buf, int n) {
    MemReader* r = static_cast<MemReader*>(op);
    const int64_t size = (int64_t)r->f->data.size();
    if (r->pos >= size) return AVERROR_EOF;
    const int k = (int)std::min<int64_t>(n, size - r->pos);
    std::memcpy(buf, r->f->data.data() + r->pos, (size_t)k);
    r->pos += k;
    return k;
}

int64_t mem_read_seek(void* op, int64_t off, int whence) {
    MemReader* r = static_cast<MemReader*>(op);
    const int64_t size = (int64_t)r->f->data.size();
    switch (whence & ~AVSEEK_FORCE) {
    case AVSEEK_SIZE: return size;
    case SEEK_SET: r->pos = off; break;
    case SEEK_CUR: r->pos += off; break;
    case SEEK_END: r->pos = size + off; break;
    default: return AVERROR(EINVAL);
    }
    return r->pos < 0 ? AVERROR(EINVAL) : r->pos;
}

constexpr int kIoBuf = 1 << 16;

std::string av_why(const char* what, int e) {
    char b[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, b, sizeof b);
    return std::string(what) + ": " + b;
}

}  /* namespace */

struct VideoWriter::Impl {
    MemFile          file;
    AVFormatContext* oc = nullptr;
    AVCodecContext*  vc = nullptr;
    AVCodecContext*  ac = nullptr;
    AVStream*        vs = nullptr;
    AVStream*        as = nullptr;
    SwsContext*      sws = nullptr;
    SwrContext*      swr = nullptr;   /* soundtrack to 48 kHz for Opus; null otherwise */
    AVFrame*         vf = nullptr;
    AVFrame*         af = nullptr;
    AVPacket*        pkt = nullptr;
    int32_t          w = 0, h = 0;
    int64_t          channels = 0;
    int64_t          next_frame = 0;
    int              aframe = 0;      /* the soundtrack frame length `af` is sized for */
    int              in_rate = 0;     /* the rate add_audio arrives at, for swr_get_delay */
    std::vector<float> audio;         /* interleaved at in_rate, encoded at finish */
    std::vector<float> audio_out;     /* interleaved at the encoder's rate after resampling */
    VideoFormat      fmt = VideoFormat::Mp4;
    bool             header = false;

    ~Impl() {
        if (oc) {
            if (oc->pb) {
                av_freep(&oc->pb->buffer);
                avio_context_free(&oc->pb);
            }
            avformat_free_context(oc);
        }
        avcodec_free_context(&vc);
        avcodec_free_context(&ac);
        sws_freeContext(sws);
        swr_free(&swr);
        av_frame_free(&vf);
        av_frame_free(&af);
        av_packet_free(&pkt);
    }

    /* Every packet `c` has ready, onto stream `st`. */
    int drain(AVCodecContext* c, AVStream* st, std::string* why) {
        for (;;) {
            const int e = avcodec_receive_packet(c, pkt);
            if (e == AVERROR(EAGAIN) || e == AVERROR_EOF) return RAD_OK;
            if (e < 0) {
                if (why) *why = av_why("encoding", e);
                return RAD_E_INVAL;
            }
            av_packet_rescale_ts(pkt, c->time_base, st->time_base);
            pkt->stream_index = st->index;
            const int w2 = av_interleaved_write_frame(oc, pkt);
            if (w2 < 0) {
                if (why) *why = av_why("writing the file", w2);
                return RAD_E_INVAL;
            }
        }
    }
};

/* The reader shift_data opens on the output's own URL; any other URL is refused. Only wired for
 * MP4 -- WebM writes forward and never reopens. */
static int open_reader(AVFormatContext* s, AVIOContext** pb, const char*, int flags, AVDictionary**) {
    if (!(flags & AVIO_FLAG_READ) || (flags & AVIO_FLAG_WRITE)) return AVERROR(EINVAL);
    MemFile* f = static_cast<MemFile*>(s->opaque);
    MemReader* r = new MemReader{ f, 0 };
    uint8_t* buf = static_cast<uint8_t*>(av_malloc(kIoBuf));
    *pb = buf ? avio_alloc_context(buf, kIoBuf, 0, r, mem_read, nullptr, mem_read_seek) : nullptr;
    if (!*pb) {
        av_free(buf);
        delete r;
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int close_reader(AVFormatContext*, AVIOContext* pb) {
    if (!pb) return 0;
    delete static_cast<MemReader*>(pb->opaque);
    av_freep(&pb->buffer);
    avio_context_free(&pb);
    return 0;
}

VideoWriter::VideoWriter() : p_(std::make_unique<Impl>()) {}
VideoWriter::~VideoWriter() = default;

int VideoWriter::open(int32_t width, int32_t height, int64_t fps_num, int64_t fps_den,
                      int64_t sample_rate, int64_t channels, std::string* why, VideoFormat fmt) {
    Impl& p = *p_;
    p.fmt = fmt;
    /* The encoders print their settings and a file's statistics at FFmpeg's info level, on stderr,
     * for every video; a failure comes back through the calls here, so only errors are let
     * through. */
    static std::once_flag quiet;
    std::call_once(quiet, [] { if (av_log_get_level() > AV_LOG_ERROR) av_log_set_level(AV_LOG_ERROR); });
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1) || fps_num <= 0 || fps_den <= 0) {
        if (why) *why = "the frames must be a positive even size at a positive rate";
        return RAD_E_INVAL;
    }
    p.w = width;
    p.h = height;
    p.channels = channels;
    const bool     mp4   = fmt == VideoFormat::Mp4;
    const char*    mux   = mp4 ? "mp4" : "webm";
    const char*    url   = mp4 ? "radiance.mp4" : "radiance.webm";
    if (avformat_alloc_output_context2(&p.oc, nullptr, mux, nullptr) < 0 || !p.oc) {
        if (why) *why = std::string("this libavformat has no ") + (mp4 ? "MP4" : "WebM") + " muxer";
        return RAD_E_UNSUPPORTED;
    }
    p.oc->url = av_strdup(url);
    p.oc->opaque = &p.file;
    if (mp4) {              /* only MP4's faststart reopens the output for reading */
        p.oc->io_open = open_reader;
        p.oc->io_close2 = close_reader;
    }
    uint8_t* buf = static_cast<uint8_t*>(av_malloc(kIoBuf));
    p.oc->pb = buf ? avio_alloc_context(buf, kIoBuf, 1, &p.file, nullptr, mem_write, mem_seek)
                   : nullptr;
    p.pkt = av_packet_alloc();
    if (!p.oc->url || !p.oc->pb || !p.pkt) {
        if (!p.oc->pb) av_free(buf);
        if (why) *why = "out of memory";
        return RAD_E_NOMEM;
    }

    /* ---- video */
    const AVCodec* vcodec = nullptr;
    if (mp4) {
        for (const char* name : { "libx264", "libopenh264" })
            if ((vcodec = avcodec_find_encoder_by_name(name))) break;
        if (!vcodec) vcodec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!vcodec) {
            if (why) *why = "this libavcodec has no H.264 encoder";
            return RAD_E_UNSUPPORTED;
        }
    } else if (fmt == VideoFormat::WebmVp9) {
        if (!(vcodec = avcodec_find_encoder_by_name("libvpx-vp9")))
            vcodec = avcodec_find_encoder(AV_CODEC_ID_VP9);
        if (!vcodec) {
            if (why) *why = "this libavcodec has no VP9 encoder";
            return RAD_E_UNSUPPORTED;
        }
    } else {   /* WebmAv1 */
        for (const char* name : { "libaom-av1", "libsvtav1" })
            if ((vcodec = avcodec_find_encoder_by_name(name))) break;
        if (!vcodec) vcodec = avcodec_find_encoder(AV_CODEC_ID_AV1);
        if (!vcodec) {
            if (why) *why = "this libavcodec has no AV1 encoder";
            return RAD_E_UNSUPPORTED;
        }
    }
    venc_name_ = vcodec->name;
    p.vs = avformat_new_stream(p.oc, nullptr);
    p.vc = avcodec_alloc_context3(vcodec);
    if (!p.vs || !p.vc) {
        if (why) *why = "out of memory";
        return RAD_E_NOMEM;
    }
    p.vc->width = width;
    p.vc->height = height;
    p.vc->pix_fmt = AV_PIX_FMT_YUV420P;
    p.vc->time_base = AVRational{ (int)fps_den, (int)fps_num };
    p.vc->framerate = AVRational{ (int)fps_num, (int)fps_den };
    p.vc->gop_size = (int)std::max<int64_t>(1, 2 * fps_num / fps_den);
    p.vc->colorspace = AVCOL_SPC_BT709;
    p.vc->color_primaries = AVCOL_PRI_BT709;
    p.vc->color_trc = AVCOL_TRC_BT709;
    p.vc->color_range = AVCOL_RANGE_MPEG;
    if (venc_name_ == "libx264") {
        /* Constant quality, visually lossless for generated footage. */
        av_opt_set(p.vc->priv_data, "preset", "medium", 0);
        av_opt_set(p.vc->priv_data, "crf", "18", 0);
    } else if (fmt == VideoFormat::WebmVp9) {
        /* Constant quality, and the speed/threading knobs that make a software VP9 usable: a row
         * of tiles at a time, a good (not realtime) deadline, alternate references for the rate. */
        p.vc->bit_rate = 0;   /* a zero bitrate + crf => constant quality for libvpx */
        av_opt_set(p.vc->priv_data, "crf", "32", 0);
        av_opt_set(p.vc->priv_data, "cpu-used", "4", 0);
        av_opt_set(p.vc->priv_data, "deadline", "good", 0);
        av_opt_set(p.vc->priv_data, "row-mt", "1", 0);
        av_opt_set(p.vc->priv_data, "auto-alt-ref", "1", 0);
    } else if (fmt == VideoFormat::WebmAv1) {
        /* libaom is slow; the realtime deadline and the highest cpu-used keep a short clip
         * finishing in reasonable time. */
        p.vc->bit_rate = 0;
        av_opt_set(p.vc->priv_data, "crf", "34", 0);
        av_opt_set(p.vc->priv_data, "cpu-used", "8", 0);
        av_opt_set(p.vc->priv_data, "deadline", "realtime", 0);
        av_opt_set(p.vc->priv_data, "row-mt", "1", 0);
        av_opt_set(p.vc->priv_data, "tiles", "2x2", 0);
    } else {
        /* An encoder without a quality mode: 0.2 bits a pixel, about 5 Mbit/s at 1344x768 x 24. */
        p.vc->bit_rate = (int64_t)((double)width * height * fps_num / fps_den * 0.2);
    }
    if (p.oc->oformat->flags & AVFMT_GLOBALHEADER) p.vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    int e = avcodec_open2(p.vc, vcodec, nullptr);
    if (e < 0) {
        if (why) *why = av_why(("opening the " + venc_name_ + " encoder").c_str(), e);
        return RAD_E_INVAL;
    }
    avcodec_parameters_from_context(p.vs->codecpar, p.vc);
    p.vs->time_base = p.vc->time_base;
    p.vs->avg_frame_rate = p.vc->framerate;
    p.vf = av_frame_alloc();
    if (!p.vf) return RAD_E_NOMEM;
    p.vf->format = AV_PIX_FMT_YUV420P;
    p.vf->width = width;
    p.vf->height = height;
    if (av_frame_get_buffer(p.vf, 0) < 0) return RAD_E_NOMEM;
    p.sws = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P,
                           SWS_BICUBIC | SWS_ACCURATE_RND, nullptr, nullptr, nullptr);
    if (!p.sws) {
        if (why) *why = "libswscale cannot convert RGB to yuv420p";
        return RAD_E_UNSUPPORTED;
    }
    /* Full-range RGB to limited-range BT.709, what the stream's colour tags say. */
    sws_setColorspaceDetails(p.sws, sws_getCoefficients(SWS_CS_DEFAULT), 1,
                             sws_getCoefficients(SWS_CS_ITU709), 0, 0, 1 << 16, 1 << 16);

    /* ---- audio: AAC at the soundtrack's own rate for MP4, Opus at 48 kHz for WebM. */
    if (channels > 0) {
        const AVSampleFormat sfmt = mp4 ? AV_SAMPLE_FMT_FLTP : AV_SAMPLE_FMT_FLT;
        const int64_t        rate = mp4 ? sample_rate : 48000;
        const AVCodec*       acodec =
            avcodec_find_encoder(mp4 ? AV_CODEC_ID_AAC : AV_CODEC_ID_OPUS);
        if (!acodec) {
            if (why) *why = std::string("this libavcodec has no ") + (mp4 ? "AAC" : "Opus") +
                            " encoder";
            return RAD_E_UNSUPPORTED;
        }
        p.as = avformat_new_stream(p.oc, nullptr);
        p.ac = avcodec_alloc_context3(acodec);
        if (!p.as || !p.ac) return RAD_E_NOMEM;
        p.ac->sample_fmt = sfmt;
        p.ac->sample_rate = (int)rate;
        av_channel_layout_default(&p.ac->ch_layout, (int)channels);
        p.ac->bit_rate = 96000 * channels;
        p.ac->time_base = AVRational{ 1, (int)rate };
        if (p.oc->oformat->flags & AVFMT_GLOBALHEADER) p.ac->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        e = avcodec_open2(p.ac, acodec, nullptr);
        if (e < 0) {
            if (why) *why = av_why(("opening the " + std::string(mp4 ? "AAC" : "Opus") +
                                    " encoder").c_str(), e);
            return RAD_E_INVAL;
        }
        avcodec_parameters_from_context(p.as->codecpar, p.ac);
        p.as->time_base = p.ac->time_base;
        p.aframe = p.ac->frame_size > 0 ? p.ac->frame_size : (mp4 ? 1024 : 960);
        p.af = av_frame_alloc();
        if (!p.af) return RAD_E_NOMEM;
        p.af->format = sfmt;
        p.af->sample_rate = p.ac->sample_rate;
        av_channel_layout_copy(&p.af->ch_layout, &p.ac->ch_layout);
        p.af->nb_samples = p.aframe;
        if (av_frame_get_buffer(p.af, 0) < 0) return RAD_E_NOMEM;
        /* Opus only encodes 8/12/16/24/48 kHz; resample any other rate to the 48 kHz it was opened
         * at. Interleaved float in and out -- the same layout the samples already sit in. */
        p.in_rate = (int)sample_rate;
        if (!mp4 && sample_rate > 0 && sample_rate != rate) {
            AVChannelLayout in_lay{}, out_lay{};
            av_channel_layout_default(&in_lay, (int)channels);
            av_channel_layout_default(&out_lay, (int)channels);
            if (swr_alloc_set_opts2(&p.swr, &out_lay, AV_SAMPLE_FMT_FLT, (int)rate, &in_lay,
                                    AV_SAMPLE_FMT_FLT, (int)sample_rate, 0, nullptr) < 0 ||
                swr_init(p.swr) < 0) {
                av_channel_layout_uninit(&in_lay);
                av_channel_layout_uninit(&out_lay);
                if (why) *why = "libswresample cannot convert the soundtrack to 48 kHz";
                return RAD_E_UNSUPPORTED;
            }
            av_channel_layout_uninit(&in_lay);
            av_channel_layout_uninit(&out_lay);
        }
    }

    AVDictionary* opts = nullptr;
    if (mp4) av_dict_set(&opts, "movflags", "+faststart", 0);
    e = avformat_write_header(p.oc, &opts);
    av_dict_free(&opts);
    if (e < 0) {
        if (why) *why = av_why(("writing the " + std::string(mp4 ? "MP4" : "WebM") +
                                " header").c_str(), e);
        return RAD_E_INVAL;
    }
    p.header = true;
    return RAD_OK;
}

int VideoWriter::add_frames(const uint8_t* rgb, int32_t n, std::string* why) {
    Impl& p = *p_;
    if (!p.header) {
        if (why) *why = "the writer is not open";
        return RAD_E_STATE;
    }
    const int64_t frame_bytes = (int64_t)p.w * p.h * 3;
    for (int32_t i = 0; i < n; ++i) {
        if (av_frame_make_writable(p.vf) < 0) return RAD_E_NOMEM;
        const uint8_t* src[1] = { rgb + i * frame_bytes };
        const int stride[1] = { p.w * 3 };
        sws_scale(p.sws, src, stride, 0, p.h, p.vf->data, p.vf->linesize);
        p.vf->pts = p.next_frame++;
        const int e = avcodec_send_frame(p.vc, p.vf);
        if (e < 0) {
            if (why) *why = av_why("encoding a frame", e);
            return RAD_E_INVAL;
        }
        RAD_TRY(p.drain(p.vc, p.vs, why));
    }
    return RAD_OK;
}

int VideoWriter::add_audio(const float* samples, int64_t n_frames, std::string* why) {
    Impl& p = *p_;
    if (!p.header || !p.ac) {
        if (why) *why = "the writer has no soundtrack";
        return RAD_E_STATE;
    }
    p.audio.insert(p.audio.end(), samples, samples + n_frames * p.channels);
    return RAD_OK;
}

int VideoWriter::finish(std::string* out, std::string* why) {
    Impl& p = *p_;
    if (!p.header) {
        if (why) *why = "the writer is not open";
        return RAD_E_STATE;
    }
    if (p.ac) {
        /* The soundtrack at the encoder's rate: resampled whole for Opus (small, a few MB at most),
         * the accumulated samples as they came for AAC. */
        const std::vector<float>* buf = &p.audio;
        if (p.swr) {
            const int64_t ch = p.channels;
            const int64_t in_n = (int64_t)p.audio.size() / ch;
            const int64_t out_cap = av_rescale_rnd(swr_get_delay(p.swr, p.in_rate) + in_n,
                                                   p.ac->sample_rate, p.in_rate, AV_ROUND_UP) + 16;
            std::vector<float> tmp((size_t)(out_cap * ch));
            const uint8_t*  in_plane[1]  = { reinterpret_cast<const uint8_t*>(p.audio.data()) };
            uint8_t*        out_plane[1] = { reinterpret_cast<uint8_t*>(tmp.data()) };
            int got = swr_convert(p.swr, out_plane, (int)out_cap, in_plane, (int)in_n);
            if (got < 0) {
                if (why) *why = av_why("resampling the soundtrack", got);
                return RAD_E_INVAL;
            }
            int64_t written = got;
            out_plane[0] = reinterpret_cast<uint8_t*>(tmp.data() + written * ch);
            int more = swr_convert(p.swr, out_plane, (int)(out_cap - written), nullptr, 0);
            if (more > 0) written += more;
            tmp.resize((size_t)(written * ch));
            p.audio_out.swap(tmp);
            buf = &p.audio_out;
        }
        /* Frames of the encoder's length, in its sample format (planar for AAC, packed for Opus);
         * the last frame padded with silence to a full frame, which Opus in particular requires and
         * which the encoder's own length bookkeeping trims. */
        const int64_t total = (int64_t)buf->size() / p.channels;
        const int     fs = p.aframe;
        const bool    planar = p.ac->sample_fmt == AV_SAMPLE_FMT_FLTP;
        for (int64_t at = 0; at < total; at += fs) {
            if (av_frame_make_writable(p.af) < 0) return RAD_E_NOMEM;
            const int64_t k = std::min<int64_t>(fs, total - at);
            if (planar) {
                for (int64_t c = 0; c < p.channels; ++c) {
                    float* d = reinterpret_cast<float*>(p.af->data[c]);
                    for (int64_t i = 0; i < k; ++i)
                        d[i] = (*buf)[(size_t)((at + i) * p.channels + c)];
                }
                p.af->nb_samples = (int)k;
            } else {
                float* d = reinterpret_cast<float*>(p.af->data[0]);
                std::memcpy(d, buf->data() + at * p.channels, (size_t)(k * p.channels * sizeof(float)));
                if (k < fs)
                    std::memset(d + k * p.channels, 0, (size_t)((fs - k) * p.channels * sizeof(float)));
                p.af->nb_samples = fs;   /* Opus: always a whole frame */
            }
            p.af->pts = at;
            const int e = avcodec_send_frame(p.ac, p.af);
            if (e < 0) {
                if (why) *why = av_why("encoding the soundtrack", e);
                return RAD_E_INVAL;
            }
            RAD_TRY(p.drain(p.ac, p.as, why));
            p.af->nb_samples = fs;
        }
        avcodec_send_frame(p.ac, nullptr);
        RAD_TRY(p.drain(p.ac, p.as, why));
    }
    avcodec_send_frame(p.vc, nullptr);
    RAD_TRY(p.drain(p.vc, p.vs, why));
    const int e = av_write_trailer(p.oc);
    if (e < 0) {
        if (why) *why = av_why(("finishing the " + std::string(p.fmt == VideoFormat::Mp4 ? "MP4"
                                                                : "WebM") + " file").c_str(), e);
        return RAD_E_INVAL;
    }
    avio_flush(p.oc->pb);
    *out = std::move(p.file.data);
    p.file.data.clear();
    p.header = false;
    return RAD_OK;
}

#endif  /* RAD_HAVE_FFMPEG */

}  /* namespace mm */
}  /* namespace rad */
