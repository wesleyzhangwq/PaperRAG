package com.wesz.paperrag.auth;

import java.util.Map;
import java.util.Optional;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
public class InMemoryUserAccountStore implements UserAccountStore {

    private final Map<String, UserAccount> users;

    public InMemoryUserAccountStore(PasswordEncoder passwordEncoder) {
        users = Map.of(
            "alice", new UserAccount("alice", passwordEncoder.encode("password"), 1L),
            "bob", new UserAccount("bob", passwordEncoder.encode("password"), 2L)
        );
    }

    @Override
    public Optional<UserAccount> findByUsername(String username) {
        return Optional.ofNullable(users.get(username));
    }
}
