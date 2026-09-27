package ru.moscow.tram.forecast.api;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/v1")
public class NowcastController {
    private final ForecastDataClient dataClient;

    public NowcastController(ForecastDataClient dataClient) {
        this.dataClient = dataClient;
    }

    @GetMapping("/nowcast")
    public Mono<NowcastResponse> nowcast(
            @RequestParam int route,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date,
            @RequestParam int now_hour) {
        if (route <= 0) {
            throw new IllegalArgumentException("Параметр route должен быть положительным");
        }
        if (now_hour < 0 || now_hour > 23) {
            throw new IllegalArgumentException("Параметр now_hour должен быть от 0 до 23");
        }
        return dataClient.nowcastInputs(route, date)
                .map(input -> calculate(input, now_hour));
    }

    private NowcastResponse calculate(ForecastDataClient.NowcastInputs input, int nowHour) {
        ForecastDataClient.NowcastConfig config = input.nowcast();
        int startHour = config == null ? 5 : config.start_hour();
        double shrink = config == null ? 0.7 : config.shrink();
        List<Double> clip = config == null ? List.of(0.5, 1.5) : config.clip();
        double clipLow = clip != null && clip.size() >= 2 ? clip.get(0) : 0.5;
        double clipHigh = clip != null && clip.size() >= 2 ? clip.get(1) : 1.5;

        Map<Integer, Integer> actualByHour = input.actuals().stream()
                .collect(Collectors.toMap(ForecastDataClient.ActualHour::hour,
                        ForecastDataClient.ActualHour::boardings, (left, right) -> right));
        Map<Integer, ForecastDataClient.BaselineHour> baselineByHour = input.baseline().stream()
                .collect(Collectors.toMap(ForecastDataClient.BaselineHour::hour, Function.identity(),
                        (left, right) -> right));

        double observed = 0;
        double expected = 0;
        if (nowHour >= startHour) {
            for (int hour = startHour; hour <= nowHour; hour++) {
                observed += actualByHour.getOrDefault(hour, 0);
                ForecastDataClient.BaselineHour baseline = baselineByHour.get(hour);
                if (baseline != null) {
                    expected += baseline.yhat();
                }
            }
        }
        double ratio = expected > 0 ? observed / expected : 1.0;
        double correction = Math.max(clipLow, Math.min(clipHigh, ratio));
        double multiplier = Math.max(0.0, 1.0 + shrink * (correction - 1.0));

        List<NowcastPoint> points = input.baseline().stream()
                .filter(point -> point.hour() > nowHour)
                .sorted((left, right) -> Integer.compare(left.hour(), right.hour()))
                .map(point -> new NowcastPoint(
                        point.hour(),
                        point.yhat(),
                        Math.max(0.0, point.yhat() * multiplier),
                        Math.max(0.0, point.yhat() * point.q10_ratio() * multiplier),
                        Math.max(0.0, point.yhat() * point.q90_ratio() * multiplier)))
                .toList();

        NowcastMeta meta = new NowcastMeta(shrink, List.of(clipLow, clipHigh), startHour,
                config == null ? null : config.gain_note());
        return new NowcastResponse(input.route(), input.date(), nowHour, correction, meta, points);
    }

    public record NowcastResponse(
            int route, LocalDate date, int now_hour, double correction_factor,
            NowcastMeta meta, List<NowcastPoint> points) {}
    public record NowcastMeta(double shrink, List<Double> clip, int start_hour, String gain_note) {}
    public record NowcastPoint(int hour, double baseline, double yhat_adj, double q10, double q90) {}
}
