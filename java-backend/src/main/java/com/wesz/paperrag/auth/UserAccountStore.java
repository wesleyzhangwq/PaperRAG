package com.wesz.paperrag.auth;

import java.util.Optional;

public interface UserAccountStore {

    Optional<UserAccount> findByUsername(String username);
}
