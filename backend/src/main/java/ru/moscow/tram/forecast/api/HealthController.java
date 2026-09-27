package ru.moscow.tram.forecast.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/v1")
public class HealthController {
    private final ForecastDataClient dataClient;

    public HealthController(ForecastDataClient dataClient) {
        this.dataClient = dataClient;
    }

    @GetMapping("/health")
    public Mono<HealthResponse> health() {
        return dataClient.health()
                .map(health -> new HealthResponse(health.status(), health.model_version(),
                        health.lb_wape_score(), health.cv_wape_score(), health.trained_at()))
                .onErrorResume(error -> Mono.just(new HealthResponse("degraded", "unavailable", null, null, null)));
    }
}
