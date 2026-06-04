package com.wesz.paperrag.auth;

import com.wesz.paperrag.common.BusinessException;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final UserAccountStore userAccountStore;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthService(
        UserAccountStore userAccountStore,
        PasswordEncoder passwordEncoder,
        JwtService jwtService
    ) {
        this.userAccountStore = userAccountStore;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    public LoginResponse login(LoginRequest request) {
        UserAccount account = userAccountStore.findByUsername(request.username())
            .filter(user -> passwordEncoder.matches(request.password(), user.passwordHash()))
            .orElseThrow(() -> new BusinessException(HttpStatus.UNAUTHORIZED, "Invalid credentials"));
        return new LoginResponse(jwtService.issue(account), account.username(), account.tenantId());
    }
}
