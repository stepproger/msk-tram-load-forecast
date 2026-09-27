package ru.moscow.tram.forecast.api;

import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/v1")
public class DataProxyController {
    private final ForecastDataClient dataClient;
    public DataProxyController(ForecastDataClient dataClient) { this.dataClient = dataClient; }

    @GetMapping("/routes")
    public Mono<ResponseEntity<byte[]>> routes() {
        return dataClient.getJson("/internal/routes", null);
    }

    @GetMapping("/routes/{route}/geometry")
    public Mono<ResponseEntity<byte[]>> geometry(@PathVariable int route) {
        return dataClient.getJson("/internal/routes/" + route + "/geometry", null);
    }

    @GetMapping({"/forecast", "/map/snapshot", "/recommendations", "/factors", "/events", "/components",
            "/plan-fact", "/anomalies", "/risk", "/export"})
    public Mono<ResponseEntity<byte[]>> getData(ServerWebExchange exchange) {
        return dataClient.getJson(internalPath(exchange), exchange.getRequest().getURI().getRawQuery());
    }

    /** Dispatcher-added network events (repair, launch, mass event) that adjust the forecast. */
    @PostMapping("/events")
    public Mono<ResponseEntity<byte[]>> addEvent(@RequestBody(required = false) byte[] body) {
        return dataClient.send(HttpMethod.POST, "/internal/events", body);
    }

    @DeleteMapping("/events/{id}")
    public Mono<ResponseEntity<byte[]>> deleteEvent(@PathVariable String id) {
        if (!id.matches("[A-Za-z0-9_-]{1,64}")) {
            throw new IllegalArgumentException("Некорректный идентификатор события");
        }
        return dataClient.send(HttpMethod.DELETE, "/internal/events/" + id, null);
    }

    private static String internalPath(ServerWebExchange exchange) {
        return exchange.getRequest().getURI().getRawPath().replaceFirst("^/api/v1", "/internal");
    }
}
