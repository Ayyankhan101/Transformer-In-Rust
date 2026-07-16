use anyhow::Result;

use crate::cli::Cli;

pub fn run(cli: &Cli, port: u16) -> Result<()> {
    #[cfg(feature = "server")]
    {
        use std::net::SocketAddr;
        use crate::server;
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(server::start_server(
            cli.weights_dir.join("pytorch_model.bin").to_str().unwrap(),
            addr,
        ))
    }

    #[cfg(not(feature = "server"))]
    {
        let _ = (cli, port);
        anyhow::bail!(
            "Server requires the 'server' feature.\n\
             Rebuild with: cargo build --features server"
        )
    }
}
