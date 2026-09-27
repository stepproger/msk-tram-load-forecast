package ru.moscow.tram.forecast.api;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.ReactiveAuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.ServerAuthenticationEntryPoint;
import org.springframework.security.web.server.context.NoOpServerSecurityContextRepository;
import reactor.core.publisher.Mono;

/**
 * HTTP Basic for the dispatcher API. The 401 response carries no WWW-Authenticate challenge,
 * so browsers do not show their native login dialog: the SPA renders its own login screen.
 */
@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {
    private static final byte[] UNAUTHORIZED_BODY =
            "{\"error\":\"unauthorized\",\"message\":\"Требуется вход: неверный логин или пароль\"}"
                    .getBytes(StandardCharsets.UTF_8);

    @Bean
    SecurityWebFilterChain apiSecurity(ServerHttpSecurity http, ReactiveAuthenticationManager dispatcherAuthentication) {
        ServerAuthenticationEntryPoint entryPoint = (exchange, exception) -> {
            var response = exchange.getResponse();
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            response.getHeaders().setContentType(MediaType.APPLICATION_JSON);
            return response.writeWith(Mono.just(response.bufferFactory().wrap(UNAUTHORIZED_BODY)));
        };
        return http
                .csrf(ServerHttpSecurity.CsrfSpec::disable)
                .formLogin(ServerHttpSecurity.FormLoginSpec::disable)
                .logout(ServerHttpSecurity.LogoutSpec::disable)
                .securityContextRepository(NoOpServerSecurityContextRepository.getInstance())
                .httpBasic(basic -> basic.authenticationManager(dispatcherAuthentication).authenticationEntryPoint(entryPoint))
                .exceptionHandling(handling -> handling.authenticationEntryPoint(entryPoint))
                .authorizeExchange(exchanges -> exchanges
                        .pathMatchers("/api/v1/health").permitAll()
                        .pathMatchers("/api/**").authenticated()
                        .anyExchange().permitAll())
                .build();
    }

    /**
     * Single dispatcher account from env. A constant-time comparison replaces the default
     * user-details flow, which would re-encode the password with bcrypt and spend ~50 ms per request.
     */
    @Bean
    ReactiveAuthenticationManager dispatcherAuthentication(
            @Value("${app.user}") String username,
            @Value("${app.password}") String password) {
        if (username.isBlank() || password.isBlank()) {
            throw new IllegalStateException("APP_USER и APP_PASSWORD должны быть заданы");
        }
        byte[] expectedUser = username.getBytes(StandardCharsets.UTF_8);
        byte[] expectedPassword = password.getBytes(StandardCharsets.UTF_8);
        List<SimpleGrantedAuthority> authorities = List.of(new SimpleGrantedAuthority("ROLE_DISPATCHER"));
        return authentication -> {
            String presentedUser = authentication.getName();
            Object credentials = authentication.getCredentials();
            boolean userMatches = MessageDigest.isEqual(presentedUser.getBytes(StandardCharsets.UTF_8), expectedUser);
            boolean passwordMatches = credentials != null
                    && MessageDigest.isEqual(credentials.toString().getBytes(StandardCharsets.UTF_8), expectedPassword);
            return userMatches && passwordMatches
                    ? Mono.just(UsernamePasswordAuthenticationToken.authenticated(presentedUser, null, authorities))
                    : Mono.error(new BadCredentialsException("Неверный логин или пароль"));
        };
    }
}
