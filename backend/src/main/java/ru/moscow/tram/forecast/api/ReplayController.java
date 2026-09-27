package ru.moscow.tram.forecast.api;

import java.time.Duration;
import java.time.LocalDate;
import java.util.Map;

import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/v1")
public class ReplayController {
    private final ForecastDataClient dataClient;

    public ReplayController(ForecastDataClient dataClient) {
        this.dataClient = dataClient;
    }

    @GetMapping(value = "/stream/replay", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<ForecastDataClient.ReplayEvent>> replay(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date,
            @RequestParam(defaultValue = "60") int speed) {
        if (speed < 1 || speed > 3600) {
            throw new IllegalArgumentException("Параметр speed должен быть от 1 до 3600");
        }
        Duration step = Duration.ofMillis(Math.max(10, 60_000L / speed));
        return dataClient.replayEvents(date)
                .flatMapMany(Flux::fromIterable)
                .delayElements(step)
                .map(event -> ServerSentEvent.builder(event)
                        .event("validation")
                        .id(event.ts() + "-" + event.route())
                        .build())
                .concatWith(Flux.just(ServerSentEvent.<ForecastDataClient.ReplayEvent>builder().event("end").build()));
    }
}
